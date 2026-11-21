"""Validate and summarize completed experiment 2 predictions."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from .common import read_jsonl, sha256_file, write_json
from .experiment2 import make_jobs, orders_for
from .score_stream import qa_scores, write_csv


def summarize(documents, orders, rows):
    jobs = make_jobs(orders)
    docs = {d["doc_id"]: d for d in documents}
    expected = {}
    for job in jobs:
        for doc_id in job["targets"]:
            for qa in docs[doc_id]["questions"]:
                key = (job["group"], job["seed"], job["step"], doc_id, qa["qid"])
                expected[key] = (job, qa)
    scored = {}
    for row in rows:
        key = tuple(row[k] for k in ("group", "seed", "step", "doc_id", "qid"))
        if key in scored or key not in expected:
            raise ValueError(f"Duplicate or unexpected prediction: {key}")
        job, qa = expected[key]
        if row["state"] != [list(x) for x in job["state"]] or row["answers"] != qa["answers"]:
            raise ValueError(f"State or references differ from plan: {key}")
        if row["is_first"] != (job["first"] == row["doc_id"]):
            raise ValueError(f"Incorrect first-document flag: {key}")
        if len(row["output_token_ids"]) != row["output_tokens"]:
            raise ValueError(f"Output token count differs: {key}")
        if row["hit_output_limit"] != (row["output_tokens"] == 256):
            raise ValueError(f"Output limit flag differs: {key}")
        score = qa_scores(row["prediction"], qa["answers"])
        scored[key] = {**row, **score}
    if set(scored) != set(expected):
        raise ValueError(f"Incomplete predictions: {len(set(expected)-set(scored))} missing")

    grouped = defaultdict(list)
    for row in scored.values():
        role = "first" if row["is_first"] else "other"
        for subset in ("all", role):
            grouped[(row["seed"], row["group"], row["step"], subset)].append(row)
            if row["seed"] is not None:
                grouped[(row["seed"], row["group"], "all", subset)].append(row)
    summary = []
    for (seed, group, step, subset), values in grouped.items():
        per_doc = defaultdict(list)
        for row in values:
            per_doc[row["doc_id"]].append(row)
        summary.append(dict(seed=seed, group=group, step=step, subset=subset,
            num_docs=len(per_doc), num_questions=len(values),
            em_micro=100*mean(r["em"] for r in values),
            f1_micro=100*mean(r["f1"] for r in values),
            em_macro=100*mean(mean(r["em"] for r in rs) for rs in per_doc.values()),
            f1_macro=100*mean(mean(r["f1"] for r in rs) for rs in per_doc.values()),
            hit_limit_rate=mean(r["hit_output_limit"] for r in values),
            mean_output_tokens=mean(r["output_tokens"] for r in values)))

    paired = []
    for seed, order in orders.items():
        for step in range(5, len(order)+1, 5):
            for doc_id in order[step-5:step]:
                for group in ("A2", "A3", "B2", "B3", "B4", "B5"):
                    pairs = [(scored[(group, seed, step, doc_id, q["qid"])],
                              scored[("A1", seed, step, doc_id, q["qid"])])
                             for q in docs[doc_id]["questions"]]
                    diff = 100*mean(a["f1"]-b["f1"] for a, b in pairs)
                    paired.append(dict(seed=seed, step=step, group=group, doc_id=doc_id,
                        position=order.index(doc_id)+1, is_first=doc_id==order[step-5],
                        num_questions=len(pairs), f1_diff=diff,
                        change="higher" if diff > 0 else "lower" if diff < 0 else "same",
                        em_diff=100*mean(a["em"]-b["em"] for a,b in pairs),
                        hit_limit_rate_diff=mean(int(a["hit_output_limit"])-int(b["hit_output_limit"]) for a,b in pairs)))
                for baseline in ("base", "reset_1", "reset_0.5", "reset_0.25"):
                    pairs = [(scored[("A1", seed, step, doc_id, q["qid"])],
                              scored[(baseline, None, 0, doc_id, q["qid"])])
                             for q in docs[doc_id]["questions"]]
                    diff = 100*mean(a["f1"]-b["f1"] for a,b in pairs)
                    paired.append(dict(seed=seed, step=step, group=f"A1_minus_{baseline}", doc_id=doc_id,
                        position=order.index(doc_id)+1, is_first=doc_id==order[step-5],
                        num_questions=len(pairs), f1_diff=diff,
                        change="higher" if diff > 0 else "lower" if diff < 0 else "same",
                        em_diff=100*mean(a["em"]-b["em"] for a,b in pairs),
                        hit_limit_rate_diff=mean(int(a["hit_output_limit"])-int(b["hit_output_limit"]) for a,b in pairs)))
            doc_id = order[step-5]
            for group in ("C0", "C0.25", "C0.5", "C1"):
                pairs = [(scored[(group, seed, step, doc_id, q["qid"])],
                          scored[("A1", seed, step, doc_id, q["qid"])])
                         for q in docs[doc_id]["questions"]]
                diff = 100*mean(a["f1"]-b["f1"] for a,b in pairs)
                paired.append(dict(seed=seed, step=step, group=group, doc_id=doc_id,
                    position=order.index(doc_id)+1, is_first=True, num_questions=len(pairs),
                    f1_diff=diff, change="higher" if diff > 0 else "lower" if diff < 0 else "same",
                    em_diff=100*mean(a["em"]-b["em"] for a,b in pairs),
                    hit_limit_rate_diff=mean(int(a["hit_output_limit"])-int(b["hit_output_limit"]) for a,b in pairs)))
    return list(scored.values()), summary, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    path = args.run_dir
    status = json.loads((path / "status.json").read_text())
    if status["status"] != "completed":
        raise ValueError("Run is incomplete")
    config = json.loads((path / "config.json").read_text())
    if sha256_file(path / "evaluation_documents.jsonl") != config["evaluation_documents_sha256"]:
        raise ValueError("Document snapshot changed")
    docs = read_jsonl(path / "evaluation_documents.jsonl")
    orders = orders_for(docs)
    if {str(k): v for k,v in orders.items()} != config["orders"]:
        raise ValueError("Orders do not match frozen snapshot")
    scored, summary, paired = summarize(docs, orders, read_jsonl(path / "predictions.jsonl"))
    write_csv(path / "scores.csv", scored)
    write_csv(path / "summary.csv", summary)
    write_csv(path / "paired_by_document.csv", paired)
    document_rows = defaultdict(list)
    for row in scored:
        document_rows[(row["seed"], row["group"], row["step"], row["doc_id"], row["is_first"])].append(row)
    write_csv(path / "document_scores.csv", [dict(seed=seed, group=group, step=step,
        doc_id=doc_id, is_first=first, num_questions=len(values),
        em=100*mean(v["em"] for v in values), f1=100*mean(v["f1"] for v in values),
        hit_limit_rate=mean(v["hit_output_limit"] for v in values),
        mean_output_tokens=mean(v["output_tokens"] for v in values))
        for (seed, group, step, doc_id, first), values in document_rows.items()])
    ranges = defaultdict(list)
    for row in summary:
        if row["seed"] is not None:
            ranges[(row["group"], row["step"], row["subset"])].append(row)
    write_csv(path / "seed_ranges.csv", [dict(group=group, step=step, subset=subset,
        seeds=len(values), f1_micro_min=min(v["f1_micro"] for v in values),
        f1_micro_max=max(v["f1_micro"] for v in values),
        f1_macro_min=min(v["f1_macro"] for v in values),
        f1_macro_max=max(v["f1_macro"] for v in values))
        for (group, step, subset), values in ranges.items()])
    write_json(path / "score_summary.json", {"logical_predictions": len(scored),
        "predictions_sha256": sha256_file(path / "predictions.jsonl"),
        "score_units": "summary and paired: 0-100; question: 0-1",
        "scope": "three orders of one fixed document set, not independent datasets"})
    print(f"Scored {len(scored)} logical predictions")


if __name__ == "__main__":
    main()
