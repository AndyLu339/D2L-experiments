"""Validate the complete immediate-answer experiment and write its score tables."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from .common import read_jsonl, sha256_file, write_json
from .immediate50 import make_jobs, orders_for
from .score_stream import qa_scores, write_csv


def score(documents, predictions, orders):
    docs = {doc["doc_id"]: doc for doc in documents}
    expected = {}
    for job in make_jobs(orders):
        for doc_id in job["targets"]:
            for qa in docs[doc_id]["questions"]:
                key = (job["group"], job["seed"], job["step"], doc_id, qa["qid"])
                if key in expected:
                    raise ValueError(f"Duplicate planned cell: {key}")
                expected[key] = (job, qa)
    scored = {}
    for row in predictions:
        key = tuple(row[k] for k in ("group", "seed", "step", "doc_id", "qid"))
        if key in scored or key not in expected:
            raise ValueError(f"Duplicate or unexpected prediction: {key}")
        job, qa = expected[key]
        if row["state"] != [list(x) for x in job["state"]] or row["answers"] != qa["answers"]:
            raise ValueError(f"State or references do not match plan: {key}")
        if len(row["output_token_ids"]) != row["output_tokens"]:
            raise ValueError(f"Output token count mismatch: {key}")
        if row["hit_output_limit"] != (row["output_tokens"] == 256):
            raise ValueError(f"Output limit flag mismatch: {key}")
        scored[key] = {**row, **qa_scores(row["prediction"], qa["answers"])}
    if set(scored) != set(expected):
        raise ValueError(f"Incomplete predictions: {len(set(expected) - set(scored))} missing")
    grouped = defaultdict(list)
    for row in scored.values():
        grouped[(row["group"], row["seed"], row["step"], row["doc_id"])].append(row)
    doc_scores = []
    for (group, seed, step, doc), rows in grouped.items():
        doc_scores.append({"group": group, "seed": seed, "step": step, "doc_id": doc,
            "num_questions": len(rows), "em": 100 * mean(r["em"] for r in rows),
            "f1": 100 * mean(r["f1"] for r in rows),
            "hit_limit_rate": mean(r["hit_output_limit"] for r in rows),
            "mean_output_tokens": mean(r["output_tokens"] for r in rows)})
    return list(scored.values()), doc_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    path = args.run_dir
    if json.loads((path / "status.json").read_text())["status"] != "completed":
        raise ValueError("Refusing to score an incomplete run")
    config = json.loads((path / "config.json").read_text())
    snapshot = path / "evaluation_documents.jsonl"
    if sha256_file(snapshot) != config["evaluation_documents_sha256"]:
        raise ValueError("Document snapshot changed")
    documents = read_jsonl(snapshot)
    orders = orders_for(documents)
    if {str(k): v for k, v in orders.items()} != config["orders"]:
        raise ValueError("Document orders changed")
    predictions = read_jsonl(path / "predictions.jsonl")
    scored, documents_scored = score(documents, predictions, orders)
    if len(scored) != config["expected_logical_predictions"]:
        raise ValueError("Unexpected number of predictions")
    write_csv(path / "scores.csv", scored)
    write_csv(path / "document_scores.csv", documents_scored)
    write_json(path / "score_summary.json", {
        "logical_predictions": len(scored),
        "predictions_sha256": sha256_file(path / "predictions.jsonl"),
        "score_units": "per-question 0–1; per-document 0–100",
        "plot_scope": "no plot yet; requested later for first 30 positions",
    })
    print(f"Validated and scored {len(scored)} predictions")


if __name__ == "__main__":
    main()
