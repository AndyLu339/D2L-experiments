"""Resume a paused immediate50 run into a new directory, preserving old results."""

import argparse
import json
from pathlib import Path
import random
import shutil
import traceback
from .common import read_jsonl, sha256_file, write_json
from .earlystop50 import execute
from .immediate50 import make_jobs, orders_for
from .preflight_experiment2 import preflight
from .run_experiment2 import ExperimentBackend
from .run_stream import JsonlSink, measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.max_ctx_chunk_len, args.max_new_tokens = 8192, 256
    args.batched, args.save_token_ids = False, True
    previous_config = json.loads((args.previous_run / "config.json").read_text())
    previous_snapshot = args.previous_run / "evaluation_documents.jsonl"
    if sha256_file(previous_snapshot) != previous_config["evaluation_documents_sha256"]:
        raise ValueError("Previous document snapshot changed")
    if sha256_file(args.checkpoint) != previous_config["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from previous run")
    docs = read_jsonl(previous_snapshot)
    orders = orders_for(docs)
    if {str(k): v for k,v in orders.items()} != previous_config["orders"]:
        raise ValueError("Previous orders changed")
    previous_path = args.previous_run / "predictions.jsonl"
    previous = read_jsonl(previous_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(previous_snapshot, args.output_dir / "evaluation_documents.jsonl")
    cfg = dict(previous_config)
    cfg.update(previous_run=str(args.previous_run), previous_predictions_sha256=sha256_file(previous_path),
               previous_prediction_count=len(previous), early_stop_consecutive=5,
               early_stop_policy="preserve previously generated answers, including those after trigger; skip future missing items",
               resumed_code="resume_earlystop50.py + earlystop50.py")
    write_json(args.output_dir / "config.json", cfg)
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    sink = JsonlSink(args.output_dir)
    for name in ("skipped", "early_stops"):
        sink.handles[name] = (args.output_dir / f"{name}.jsonl").open("w", encoding="utf-8")
    write_json(args.output_dir / "status.json", {"status": "running"})
    try:
        random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        with torch.inference_mode():
            with measure(sink, "load_model", {}): backend = ExperimentBackend(args, sink)
            if backend.configuration() != previous_config["model"]:
                raise ValueError("Model configuration differs from previous run")
            write_json(args.output_dir / "preflight.json", preflight(backend, docs))
            backend.describe_documents(docs); backend.encode_all(docs)
            fp = backend.raw_fingerprints()
            if fp != previous_config["raw_adapter_fingerprints"]:
                raise ValueError("Re-encoded adapter fingerprints differ from previous run")
            execute(backend, docs, make_jobs(orders), previous, sink)
            if backend.raw_fingerprints() != fp:
                raise ValueError("Cached raw adapters changed during run")
        write_json(args.output_dir / "status.json", {"status": "completed"})
    except BaseException as exc:
        write_json(args.output_dir / "status.json", {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        sink.close()


if __name__ == "__main__": main()
