"""Run the 20-document, three-order weighted LoRA experiment."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import random
import subprocess
import traceback

from .common import load_stream, sha256_file, write_json
from .experiment2 import make_jobs, orders_for, pilot_jobs, run_jobs
from .preflight_experiment2 import preflight
from .run_stream import D2LBackend, JsonlSink, measure


class ExperimentBackend(D2LBackend):
    def encode_all(self, documents):
        from .adapters import adapter_fingerprint

        self.raw = {}
        for doc in documents:
            self.reset()
            self.location = {"group": "encode", "doc_id": doc["doc_id"]}
            raw = self.encode(doc)
            chunks = next(iter(raw.values()))["A"].shape[0]
            if chunks != 1:
                raise ValueError(f"Experiment requires one chunk per document: {doc['doc_id']} has {chunks}")
            # Keep the original immutable and off GPU until the next binding.
            self.raw[doc["doc_id"]] = {m: {k: v.detach().cpu() for k, v in p.items()}
                                       for m, p in raw.items()}
            self.emit("documents", {"doc_id": doc["doc_id"], "num_chunks": chunks,
                                    "adapter_sha256": adapter_fingerprint(self.raw[doc["doc_id"]])})
        self.reset()

    def raw_fingerprints(self):
        from .adapters import adapter_fingerprint
        return {doc_id: adapter_fingerprint(raw) for doc_id, raw in self.raw.items()}

    def bind_weighted(self, state):
        self.reset()
        if not state:
            return
        with measure(self.emit, "weighted_bind", self.location):
            for doc_id, coefficient in state:
                source = self.raw[doc_id]
                weighted = {m: {"A": pair["A"].to(self.model.device) * coefficient,
                                "B": pair["B"].to(self.model.device)}
                            for m, pair in source.items()}
                self.adapters.append(weighted)
            self.adapters.bind()
        self.emit("states", {**self.location, "state": state,
                              "head_bias_once": bool(self.model.hypernet.config.use_bias),
                              **self.state_info()})


def model_file_hashes(model_name, revision):
    """Record locally resolved model weight files; never access the network."""
    from huggingface_hub import snapshot_download

    directory = Path(model_name)
    if not directory.is_dir():
        try:
            directory = Path(snapshot_download(repo_id=model_name, revision=revision,
                                               local_files_only=True))
        except Exception as exc:
            return {"resolved": False, "reason": str(exc)}
    index = directory / "model.safetensors.index.json"
    bin_index = directory / "pytorch_model.bin.index.json"
    if index.exists() or bin_index.exists():
        index = index if index.exists() else bin_index
        weights = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        files = [index] + [directory / name for name in weights]
    elif (directory / "model.safetensors").exists():
        files = [directory / "model.safetensors"]
    elif (directory / "pytorch_model.bin").exists():
        files = [directory / "pytorch_model.bin"]
    else:
        return {"resolved": False, "directory": str(directory), "reason": "no recognized weight files"}
    if any(not path.is_file() for path in files):
        return {"resolved": False, "directory": str(directory), "reason": "weight shards missing"}
    return {"resolved": True, "directory": str(directory),
            "sha256": {path.name: sha256_file(path) for path in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true", help="Freeze data and count work without GPU")
    parser.add_argument("--preflight-only", action="store_true", help="Run real-model checks without the full evaluation")
    parser.add_argument("--pilot", action="store_true", help="Run only seed 11's first five documents")
    parser.add_argument("--max-ctx-chunk-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.plan_only and args.preflight_only:
        parser.error("Choose at most one of --plan-only and --preflight-only")
    if args.pilot and args.preflight_only:
        parser.error("Choose at most one of --pilot and --preflight-only")
    if args.max_ctx_chunk_len != 8192 or args.max_new_tokens != 256:
        parser.error("Experiment 2 fixes chunk length 8192 and output budget 256")
    args.batched = False
    args.save_token_ids = True
    manifest, documents = load_stream(args.stream_dir, 11, 20)
    if manifest["dataset"] != "squad_v1.1":
        raise ValueError("Only SQuAD 1.1 is supported")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    orders = orders_for(documents)
    jobs = make_jobs(orders)
    if args.pilot:
        jobs = pilot_jobs(jobs, orders[11])
    by_id = {d["doc_id"]: d for d in documents}
    expected = sum(len(by_id[d]["questions"]) for job in jobs for d in job["targets"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshot = args.output_dir / "evaluation_documents.jsonl"
    with snapshot.open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    config = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "source_manifest_sha256": sha256_file(args.stream_dir / "manifest.json"),
              "source_documents_sha256": manifest["documents_sha256"],
              "evaluation_documents_sha256": sha256_file(snapshot),
              "checkpoint_sha256": sha256_file(args.checkpoint),
              "orders": {str(k): v for k, v in orders.items()},
              "shuffle": "Python random.Random(seed).shuffle on seed-11 prefix",
              "expected_logical_predictions": expected,
              "jobs": jobs}
    source_dir = Path(__file__).resolve().parent
    source_files = [source_dir / name for name in (
        "experiment2.py", "preflight_experiment2.py", "run_experiment2.py", "run_stream.py",
        "adapters.py", "score_experiment2.py", "score_stream.py", "common.py")]
    source_files += [source_dir.parents[1] / "src/ctx_to_lora/modeling" / name
                     for name in ("hypernet.py", "lora_merger.py", "lora_layer.py")]
    config["source_sha256"] = {str(path): sha256_file(path) for path in source_files}
    write_json(args.output_dir / "config.json", config)
    if args.plan_only:
        write_json(args.output_dir / "status.json", {"status": "planned"})
        print(f"Planned {len(jobs)} jobs, {expected} logical predictions")
        return
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        write_json(args.output_dir / "status.json", {"status": "failed", "error": "CUDA BF16 required"})
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    packages = {}
    for name in ("torch", "transformers", "peft", "datasets", "flash-attn", "bitsandbytes", "einops"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    config.update(packages=packages, cuda_version=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(),
                  code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  code_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()))
    sink = JsonlSink(args.output_dir)
    write_json(args.output_dir / "status.json", {"status": "running"})
    try:
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.inference_mode():
            with measure(sink, "load_model", {}):
                backend = ExperimentBackend(args, sink)
            config["model"] = backend.configuration()
            config["base_model_revision"] = getattr(backend.model.base_model.config, "_commit_hash", None)
            config["ctx_model_revision"] = getattr(backend.model.ctx_encoder.base_model.config, "_commit_hash", None)
            base_name = backend.model.base_model.name_or_path
            ctx_name = backend.model.ctx_encoder.base_model.name_or_path
            config["model_weight_files"] = {
                base_name: model_file_hashes(base_name, config["base_model_revision"])}
            if ctx_name != base_name:
                config["model_weight_files"][ctx_name] = model_file_hashes(ctx_name, config["ctx_model_revision"])
            write_json(args.output_dir / "config.json", config)
            write_json(args.output_dir / "preflight.json", preflight(backend, documents))
            if args.preflight_only:
                write_json(args.output_dir / "status.json", {"status": "preflight_passed"})
                return
            active_documents = documents[:5] if args.pilot else documents
            backend.describe_documents(active_documents)
            backend.encode_all(active_documents)
            fingerprints = backend.raw_fingerprints()
            config["raw_adapter_fingerprints"] = fingerprints
            write_json(args.output_dir / "config.json", config)
            run_jobs(backend, documents, jobs, sink)
            if backend.raw_fingerprints() != fingerprints:
                raise AssertionError("Cached original adapters changed during evaluation")
        write_json(args.output_dir / "status.json", {"status": "pilot_completed" if args.pilot else "completed"})
    except BaseException as exc:
        write_json(args.output_dir / "status.json", {"status": "failed", "error": str(exc),
                                                     "traceback": traceback.format_exc()})
        raise
    finally:
        sink.close()


if __name__ == "__main__":
    main()
