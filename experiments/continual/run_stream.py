"""Run Base / ICL / Reset-D2L / cumulative D2L on a frozen SQuAD stream."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import traceback

from .common import load_stream, sha256_file, write_json
from .protocol import run_protocol


class JsonlSink:
    def __init__(self, directory):
        self.handles = {
            name: (directory / f"{name}.jsonl").open("w", encoding="utf-8")
            for name in ("predictions", "resources", "states", "documents")
        }

    def __call__(self, name, record):
        self.handles[name].write(json.dumps(record, ensure_ascii=False) + "\n")
        self.handles[name].flush()
        if name == "states":
            print(
                f"{record['group']} step={record['step']} chunks={record['num_chunks']} "
                f"rank={record['physical_rank']}",
                flush=True,
            )

    def close(self):
        for handle in self.handles.values():
            handle.close()


@contextmanager
def measure(emit, stage, location, **extra):
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_bytes = torch.cuda.memory_allocated()
    start = time.perf_counter()
    record = {"stage": stage, **location, **extra, "completed": False}
    try:
        yield record
        record["completed"] = True
    finally:
        torch.cuda.synchronize()
        record.update(
            seconds=time.perf_counter() - start,
            start_allocated_bytes=start_bytes,
            end_allocated_bytes=torch.cuda.memory_allocated(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            extra_peak_allocated_bytes=torch.cuda.max_memory_allocated() - start_bytes,
        )
        emit("resources", record)


class D2LBackend:
    def __init__(self, args, emit):
        import torch
        from ctx_to_lora.data.definitions import EVAL_INTX_TEMPLATES
        from ctx_to_lora.model_loading import get_tokenizer
        from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

        from .adapters import CumulativeLoRA

        self.args, self.emit, self.location = args, emit, {}
        self.prompt_template = EVAL_INTX_TEMPLATES["squad"]
        # Public D2L checkpoints contain configuration classes as well as tensors.
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        self.model = ModulatedPretrainedModel.from_state_dict(
            state,
            train=False,
            use_sequence_packing=False,
            use_flash_attn=True,
            user_defined_scaling=1.0,
        )
        del state
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.enable_iterative_mode(not args.batched)
        if self.model.hypernet.extra_modules:
            raise ValueError("This runner supports LoRA-only checkpoints")
        self.tokenizer = get_tokenizer(self.model.base_model.name_or_path)
        self.ctx_tokenizer = get_tokenizer(
            self.model.ctx_encoder.base_model.name_or_path
        )
        self.adapters = CumulativeLoRA(self.model)
        self.reset()

    def reset(self):
        self.adapters.clear()

    def tokenize_document(self, doc):
        from ctx_to_lora.data.processing import split_too_long_ctx, tokenize_ctx_text

        ids = tokenize_ctx_text({"context": [doc["context"]]}, self.ctx_tokenizer)[
            "ctx_ids"
        ][0]
        result = split_too_long_ctx(
            {"ctx_ids": ids},
            model_name_or_path=self.ctx_tokenizer.name_or_path,
            num_chunk_probs=None,
            max_chunk_len=self.args.max_ctx_chunk_len,
            min_chunk_len=-1,
            max_num_split=None,
            is_train=False,
        )
        chunks = result["ctx_ids"]
        limit = self.model.ctx_encoder.config.max_position_embeddings
        if max(map(len, chunks)) > limit:
            raise ValueError(
                "A context chunk exceeds encoder capacity including affixes"
            )
        return chunks, {
            "doc_id": doc["doc_id"],
            "context_token_count": len(ids),
            "num_chunks": len(chunks),
            "chunk_lengths": list(map(len, chunks)),
            "question_count": len(doc["questions"]),
        }

    def describe_documents(self, documents):
        for doc in documents:
            _, metadata = self.tokenize_document(doc)
            self.emit("documents", metadata)

    def encode(self, doc):
        import torch

        with measure(
            self.emit, "encode", self.location, doc_id=doc["doc_id"]
        ) as record:
            chunks, metadata = self.tokenize_document(doc)
            record.update(metadata)
            # Serial chunks bound encoding memory; no old text or test QA is read.
            parts = []
            for chunk in chunks:
                ids = torch.tensor([chunk], device=self.model.device)
                raw, layernorms = self.model.generate_weights(ids, torch.ones_like(ids))
                if layernorms:
                    raise ValueError("Generated layernorm updates are unsupported")
                parts.append(raw)
            return {
                m: {k: torch.cat([p[m][k] for p in parts], dim=0) for k in ("A", "B")}
                for m in parts[0]
            }

    def update(self, delta):
        with measure(self.emit, "append_and_bind", self.location):
            self.adapters.append(delta)
            self.adapters.bind()

    def state_info(self):
        return self.adapters.info()

    def answer(self, doc, question, include_context=False):
        import torch

        prompt = self.prompt_template.format(input=question)
        if include_context:
            prompt = doc["context"].strip() + "\n\n" + prompt
        # Each call is a fresh single turn. No gold answers, document ID or KV
        # cache from any earlier question is passed to generate().
        ids = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
            add_generation_prompt=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)
        limit = self.model.base_model.config.max_position_embeddings
        if ids.shape[-1] + self.args.max_new_tokens > limit:
            raise ValueError(
                "Prompt plus output budget exceeds model capacity; no silent truncation"
            )
        with measure(
            self.emit,
            "generate",
            self.location,
            doc_id=doc["doc_id"],
            input_tokens=ids.shape[-1],
        ) as record:
            output = self.model.base_model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=self.args.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=False,
            )
            tokens = output[0, ids.shape[-1] :]
            record["output_tokens"] = len(tokens)
        result = {
            "prediction": self.tokenizer.decode(tokens, skip_special_tokens=True),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_tokens": ids.shape[-1],
            "output_tokens": len(tokens),
            "hit_output_limit": len(tokens) == self.args.max_new_tokens,
            "seconds": record["seconds"],
        }
        if getattr(self.args, "save_token_ids", False):
            result["output_token_ids"] = tokens.tolist()
        return result

    def configuration(self):
        return {
            "base_model": self.model.base_model.name_or_path,
            "target_dtype": str(self.model.base_model.dtype),
            "ctx_encoder_4bit": self.model.ctx_encoder_args.quantize_ctx_encoder,
            "target_modules": sorted(self.model.peft_config.target_modules),
            "lora_alpha": self.model.peft_config.lora_alpha,
            "base_rank": self.model.peft_config.r,
            "head_bias_once": self.model.hypernet.config.use_bias,
            "chunk_encoding": "serial",
            "lora_generation": "batched" if self.args.batched else "iterative",
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--order-seed", type=int, default=11)
    parser.add_argument("--num-docs", type=int, default=5)
    parser.add_argument(
        "--checkpoints", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100]
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["base", "icl", "reset", "seq"],
        default=["base", "icl", "reset", "seq"],
    )
    parser.add_argument("--max-ctx-chunk-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--batched", action="store_true", help="Batch LoRA layers (not documents)"
    )
    args = parser.parse_args()
    if args.max_ctx_chunk_len <= 0 or args.max_new_tokens <= 0:
        parser.error("Token budgets must be positive")
    if len(args.groups) != len(set(args.groups)) or any(
        c <= 0 for c in args.checkpoints
    ):
        parser.error("Groups must be unique and checkpoints positive")
    return args


def main():
    args = parse_args()
    manifest, documents = load_stream(args.stream_dir, args.order_seed, args.num_docs)
    if manifest["dataset"] != "squad_v1.1":
        raise ValueError("Only SQuAD 1.1 streams are supported")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; run the evaluator in the GPU environment"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("A GPU with BF16 support is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sink = JsonlSink(args.output_dir)
    checkpoints = sorted(
        {c for c in args.checkpoints if c <= len(documents)} | {len(documents)}
    )
    packages = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "datasets",
        "flash-attn",
        "bitsandbytes",
        "einops",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = {
        "schema_version": 1,
        "dataset": manifest["dataset"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "code_commit": commit,
        "code_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest": manifest,
        "doc_ids": [doc["doc_id"] for doc in documents],
        "num_docs": len(documents),
        "checkpoints": checkpoints,
        "groups": args.groups,
        "order_seed": args.order_seed,
        "packages": packages,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    # A frozen local snapshot makes scoring independent of later source edits.
    with (args.output_dir / "evaluation_documents.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for doc in documents:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    config["evaluation_documents_sha256"] = sha256_file(
        args.output_dir / "evaluation_documents.jsonl"
    )
    write_json(args.output_dir / "config.json", config)
    write_json(args.output_dir / "status.json", {"status": "running"})
    try:
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.inference_mode():
            with measure(sink, "load_model", {}):
                backend = D2LBackend(args, sink)
            config["model"] = backend.configuration()
            write_json(args.output_dir / "config.json", config)
            backend.describe_documents(documents)
            run_protocol(backend, documents, checkpoints, args.groups, sink)
        write_json(args.output_dir / "status.json", {"status": "completed"})
    except BaseException as exc:
        write_json(
            args.output_dir / "status.json",
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        sink.close()


if __name__ == "__main__":
    main()
