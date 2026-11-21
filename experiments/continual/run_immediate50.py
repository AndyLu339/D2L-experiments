"""Evaluate 50 documents immediately after each update for four new orders."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import random
import subprocess
import traceback

from .common import load_stream, sha256_file, write_json
from .experiment2 import run_jobs
from .immediate50 import make_jobs, orders_for
from .preflight_experiment2 import preflight
from .run_experiment2 import ExperimentBackend, model_file_hashes
from .run_stream import JsonlSink, measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.max_ctx_chunk_len = 8192
    args.max_new_tokens = 256
    args.batched = False
    args.save_token_ids = True
    manifest, documents = load_stream(args.stream_dir, 11, 50)
    if manifest["dataset"] != "squad_v1.1":
        raise ValueError("Only SQuAD 1.1 is supported")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    orders = orders_for(documents)
    jobs = make_jobs(orders)
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
              "selected_document_count": len(documents),
              "selected_question_count": sum(len(d["questions"]) for d in documents),
              "orders": {str(k): v for k, v in orders.items()},
              "shuffle": "Python random.Random(seed).shuffle on seed-11 first 50",
              "evaluation": "current document after each one-document update",
              "plot_scope": "first 30 positions, when requested; all 50 retained",
              "expected_logical_predictions": expected, "jobs": jobs}
    source_dir = Path(__file__).resolve().parent
    source_files = [source_dir / name for name in (
        "immediate50.py", "run_immediate50.py", "run_immediate50_compat.py", "score_immediate50.py",
        "experiment2.py", "preflight_experiment2.py", "run_experiment2.py", "run_stream.py",
        "adapters.py", "score_stream.py", "common.py")]
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
            config["model_weight_files"] = {base_name: model_file_hashes(base_name, config["base_model_revision"])}
            if ctx_name != base_name:
                config["model_weight_files"][ctx_name] = model_file_hashes(ctx_name, config["ctx_model_revision"])
            write_json(args.output_dir / "config.json", config)
            write_json(args.output_dir / "preflight.json", preflight(backend, documents))
            backend.describe_documents(documents)
            backend.encode_all(documents)
            fingerprints = backend.raw_fingerprints()
            config["raw_adapter_fingerprints"] = fingerprints
            write_json(args.output_dir / "config.json", config)
            run_jobs(backend, documents, jobs, sink)
            if backend.raw_fingerprints() != fingerprints:
                raise AssertionError("Cached original adapters changed during evaluation")
        write_json(args.output_dir / "status.json", {"status": "completed"})
    except BaseException as exc:
        write_json(args.output_dir / "status.json", {"status": "failed", "error": str(exc),
                                                     "traceback": traceback.format_exc()})
        raise
    finally:
        sink.close()


if __name__ == "__main__":
    main()
