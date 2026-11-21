"""Validate early-stopped runs; incomplete documents never receive a score."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
from .common import read_jsonl, sha256_file, write_json
from .immediate50 import make_jobs, orders_for
from .score_stream import qa_scores, write_csv


def score(documents, predictions, skipped, orders):
    docs = {d["doc_id"]: d for d in documents}
    expected = {}
    for job in make_jobs(orders):
        for did in job["targets"]:
            for qa in docs[did]["questions"]:
                expected[(job["group"], job["seed"], job["step"], did, qa["qid"])] = (job, qa)
    seen, grouped, scored = set(), defaultdict(list), []
    for row in predictions:
        key = tuple(row[k] for k in ("group", "seed", "step", "doc_id", "qid"))
        if key in seen or key not in expected:
            raise ValueError(f"Duplicate/unexpected prediction: {key}")
        seen.add(key)
        job, qa = expected[key]
        if row["state"] != [list(x) for x in job["state"]] or row["answers"] != qa["answers"]:
            raise ValueError(f"State or answer mismatch: {key}")
        if len(row["output_token_ids"]) != row["output_tokens"] or row["hit_output_limit"] != (row["output_tokens"] == 256):
            raise ValueError(f"Token metadata mismatch: {key}")
        record = {**row, **qa_scores(row["prediction"], qa["answers"])}
        scored.append(record)
        grouped[key[:-1]].append(record)
    skip_groups = defaultdict(int)
    for row in skipped:
        key = tuple(row[k] for k in ("group", "seed", "step", "doc_id", "qid"))
        if key in seen or key not in expected or row["reason"] != "five_consecutive_output_limits":
            raise ValueError(f"Duplicate/unexpected skip: {key}")
        seen.add(key)
        skip_groups[key[:-1]] += 1
    if seen != set(expected):
        raise ValueError(f"Incomplete accounting: {len(set(expected)-seen)} missing")
    result = []
    for key in set(grouped) | set(skip_groups):
        rows = grouped[key]
        complete = len(rows) == len(docs[key[-1]]["questions"])
        result.append(dict(zip(("group", "seed", "step", "doc_id"), key),
            expected_questions=len(docs[key[-1]]["questions"]), answered=len(rows),
            skipped=skip_groups[key], complete=complete,
            em=100*mean(r["em"] for r in rows) if complete else None,
            f1=100*mean(r["f1"] for r in rows) if complete else None,
            hit_limit_rate=mean(r["hit_output_limit"] for r in rows) if complete else None,
            previous_after_trigger=any(r["previous_after_trigger"] for r in rows)))
    return scored, result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-dir", type=Path, required=True)
    path = p.parse_args().run_dir
    if json.loads((path / "status.json").read_text())["status"] != "completed":
        raise ValueError("Refusing to score an incomplete run")
    config = json.loads((path / "config.json").read_text())
    snap = path / "evaluation_documents.jsonl"
    if sha256_file(snap) != config["evaluation_documents_sha256"]:
        raise ValueError("Document snapshot changed")
    docs = read_jsonl(snap); orders = orders_for(docs)
    if {str(k): v for k,v in orders.items()} != config["orders"]:
        raise ValueError("Orders changed")
    scored, doc = score(docs, read_jsonl(path / "predictions.jsonl"), read_jsonl(path / "skipped.jsonl"), orders)
    write_csv(path / "scores.csv", scored); write_csv(path / "document_scores.csv", doc)
    write_json(path / "score_summary.json", {"predictions": len(scored), "skipped": sum(r["skipped"] for r in doc),
        "incomplete_documents": sum(not r["complete"] for r in doc), "predictions_sha256": sha256_file(path / "predictions.jsonl")})
    print(f"Scored {len(scored)} answers; skipped cells remain missing")


if __name__ == "__main__": main()
