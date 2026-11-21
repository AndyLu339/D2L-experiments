"""Score complete SQuAD runs using official-style EM/F1 normalization."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import re
import statistics
import string

from .common import read_jsonl, sha256_file, write_json


def normalize_answer(text):
    text = "".join(char for char in text.lower() if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def qa_scores(prediction, answers):
    if not answers:
        raise ValueError("SQuAD 1.1 requires reference answers")
    normalized = normalize_answer(prediction)
    predicted_tokens = normalized.split()
    exact, f1s = [], []
    for answer in answers:
        gold = normalize_answer(answer)
        gold_tokens = gold.split()
        exact.append(float(normalized == gold))
        overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
        f1s.append(
            0.0
            if overlap == 0
            else 2 * overlap / (len(predicted_tokens) + len(gold_tokens))
        )
    return {"em": max(exact), "f1": max(f1s)}


def score_predictions(documents, predictions, groups, checkpoints):
    """Reject duplicates and missing cells instead of silently averaging partial runs."""
    n = len(documents)
    expected, references = set(), {}
    for position, doc in enumerate(documents, 1):
        for qa in doc["questions"]:
            identity = (doc["doc_id"], qa["qid"])
            if identity in references:
                raise ValueError("Duplicate question in evaluation snapshot")
            references[identity] = qa["answers"]
            for group in groups:
                phase = "immediate" if group == "seq" else "independent"
                expected.add((group, phase, position, *identity))
            if "seq" in groups:
                for checkpoint in checkpoints:
                    if checkpoint >= position:
                        expected.add(("seq", "retention", checkpoint, *identity))
    seen, scored = set(), []
    grouped = defaultdict(list)
    for row in predictions:
        key = tuple(row[k] for k in ("group", "phase", "step", "doc_id", "qid"))
        if key in seen:
            raise ValueError(f"Duplicate prediction: {key}")
        if key not in expected:
            raise ValueError(f"Unexpected prediction: {key}")
        seen.add(key)
        scores = qa_scores(row["prediction"], references[key[-2:]])
        scored.append({**row, **scores})
        grouped[key[:-1]].append(scores)
    if seen != expected:
        raise ValueError(
            f"Incomplete predictions: {len(expected - seen)} missing cells"
        )
    if "seq" in groups and n not in checkpoints:
        raise ValueError("Final retention checkpoint is required")
    document_scores = {
        key: {
            "question_count": len(scores),
            **{
                metric: statistics.mean(row[metric] for row in scores)
                for metric in ("em", "f1")
            },
        }
        for key, scores in grouped.items()
    }
    return scored, document_scores


def weighted_mean(values, weights):
    if not values:
        return None
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def paired_interval(documents, differences, weights, samples, seed):
    """Conditional paired bootstrap clustered by SQuAD source title."""
    if samples <= 0:
        return None, None
    clusters = defaultdict(list)
    for index, doc in enumerate(documents):
        title = tuple(title for title in doc["titles"] if title)
        clusters[title or (doc["doc_id"],)].append(index)
    keys = sorted(clusters)
    rng, estimates = random.Random(seed), []
    for _ in range(samples):
        indices = [
            index for key in rng.choices(keys, k=len(keys)) for index in clusters[key]
        ]
        estimates.append(
            weighted_mean(
                [differences[i] for i in indices], [weights[i] for i in indices]
            )
        )
    estimates.sort()

    def quantile(fraction):
        position = fraction * (len(estimates) - 1)
        low, high = math.floor(position), math.ceil(position)
        return estimates[low] + (estimates[high] - estimates[low]) * (position - low)

    return quantile(0.025), quantile(0.975)


def summarize(documents, scores, groups, checkpoints, bootstrap_samples=2000):
    n = len(documents)
    rows, retention = [], []
    for metric in ("em", "f1"):
        for aggregation in ("document_macro", "question_micro"):
            weights = [
                1 if aggregation == "document_macro" else len(doc["questions"])
                for doc in documents
            ]

            def values(group, phase):
                if group not in groups:
                    return None
                return [
                    scores[
                        (group, phase, n if phase == "retention" else i, doc["doc_id"])
                    ][metric]
                    for i, doc in enumerate(documents, 1)
                ]

            def average(data):
                return None if data is None else 100 * weighted_mean(data, weights)

            reset = values("reset", "independent")
            online = values("seq", "immediate")
            final = values("seq", "retention")
            row = {
                "metric": metric,
                "aggregation": aggregation,
                "num_documents": n,
                "num_questions": sum(len(doc["questions"]) for doc in documents),
                "base": average(values("base", "independent")),
                "icl": average(values("icl", "independent")),
                "reset": average(reset),
                "seq_online": average(online),
                "seq_final": average(final),
                "immediate_gap": None,
                "final_gap": None,
                "bwt": None,
                "checkpoint_forgetting": None,
            }
            for name, data in (("immediate_gap", online), ("final_gap", final)):
                row[f"{name}_ci_low"], row[f"{name}_ci_high"] = None, None
                if reset is not None and data is not None:
                    differences = [a - b for a, b in zip(data, reset)]
                    row[name] = 100 * weighted_mean(differences, weights)
                    low, high = paired_interval(
                        documents, differences, weights, bootstrap_samples, seed=42
                    )
                    row[f"{name}_ci_low"] = None if low is None else 100 * low
                    row[f"{name}_ci_high"] = None if high is None else 100 * high
            if online is not None and n > 1:
                row["bwt"] = 100 * weighted_mean(
                    [a - b for a, b in zip(final[:-1], online[:-1])], weights[:-1]
                )
                forgetting = []
                for i, doc in enumerate(documents[:-1], 1):
                    observed = [online[i - 1]] + [
                        scores[("seq", "retention", checkpoint, doc["doc_id"])][metric]
                        for checkpoint in checkpoints
                        if checkpoint >= i
                    ]
                    forgetting.append(max(observed) - final[i - 1])
                row["checkpoint_forgetting"] = 100 * weighted_mean(
                    forgetting, weights[:-1]
                )
            rows.append(row)
            if online is not None:
                for checkpoint in checkpoints:
                    data = [
                        scores[("seq", "retention", checkpoint, doc["doc_id"])][metric]
                        for doc in documents[:checkpoint]
                    ]
                    score = 100 * weighted_mean(data, weights[:checkpoint])
                    reference = (
                        None
                        if reset is None
                        else 100
                        * weighted_mean(reset[:checkpoint], weights[:checkpoint])
                    )
                    retention.append(
                        {
                            "metric": metric,
                            "aggregation": aggregation,
                            "step": checkpoint,
                            "seq_seen": score,
                            "reset_seen": reference,
                            "gap": None if reference is None else score - reference,
                        }
                    )
    return rows, retention


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_resources(resources):
    grouped = defaultdict(list)
    for record in resources:
        if not record["completed"]:
            raise ValueError("Incomplete resource stage in a completed run")
        grouped[(record["stage"], record.get("group", "load"))].append(record)
    return [
        {
            "stage": stage,
            "group": group,
            "count": len(records),
            "median_seconds": statistics.median(row["seconds"] for row in records),
            "total_seconds": sum(row["seconds"] for row in records),
            "peak_allocated_gib": max(row["peak_allocated_bytes"] for row in records)
            / 2**30,
            "peak_reserved_gib": max(row["peak_reserved_bytes"] for row in records)
            / 2**30,
            "max_extra_allocated_gib": max(
                row["extra_peak_allocated_bytes"] for row in records
            )
            / 2**30,
        }
        for (stage, group), records in sorted(grouped.items())
    ]


def plot_results(directory, documents, document_scores, retention):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    points = [
        row
        for row in retention
        if row["metric"] == "f1" and row["aggregation"] == "document_macro"
    ]
    if not points:
        return
    directory.mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(
        [p["step"] for p in points],
        [p["seq_seen"] for p in points],
        "o-",
        label="Seq: all seen",
    )
    if all(p["reset_seen"] is not None for p in points):
        ax.plot(
            [p["step"] for p in points],
            [p["reset_seen"] for p in points],
            "o--",
            label="Reset: same docs",
        )
    ax.set(
        xlabel="Documents internalized",
        ylabel="F1 (document macro, 0-100)",
        ylim=(0, 100),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "retention.png", dpi=160)
    plt.close(fig)
    matrix = [[math.nan] * len(documents) for _ in documents]
    for (group, phase, step, doc_id), scores in document_scores.items():
        if group == "seq":
            column = next(
                i for i, doc in enumerate(documents) if doc["doc_id"] == doc_id
            )
            matrix[step - 1][column] = 100 * scores["f1"]
    fig, ax = plt.subplots()
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=100, interpolation="nearest")
    ax.set(xlabel="Document position (zero-based)", ylabel="Update step (zero-based)")
    fig.colorbar(image, ax=ax, label="F1 (unmeasured cells blank)")
    fig.tight_layout()
    fig.savefig(directory / "score_matrix.png", dpi=160)
    plt.close(fig)


def score_run(directory, bootstrap_samples=2000, plots=False):
    directory = Path(directory)
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise ValueError("Run is not completed; refusing to report a partial score")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    snapshot = directory / "evaluation_documents.jsonl"
    if sha256_file(snapshot) != config["evaluation_documents_sha256"]:
        raise ValueError("Evaluation document snapshot changed")
    documents = read_jsonl(snapshot)
    if [doc["doc_id"] for doc in documents] != config["doc_ids"]:
        raise ValueError("Document order differs from run configuration")
    scored, document_scores = score_predictions(
        documents,
        read_jsonl(directory / "predictions.jsonl"),
        config["groups"],
        config["checkpoints"],
    )
    summary, retention = summarize(
        documents,
        document_scores,
        config["groups"],
        config["checkpoints"],
        bootstrap_samples,
    )
    write_csv(directory / "scores.csv", scored)
    write_csv(
        directory / "document_scores.csv",
        [
            dict(zip(("group", "phase", "step", "doc_id"), key), **scores)
            for key, scores in document_scores.items()
        ],
    )
    write_csv(directory / "summary.csv", summary)
    write_csv(directory / "retention.csv", retention)
    write_csv(
        directory / "resource_summary.csv",
        summarize_resources(read_jsonl(directory / "resources.jsonl")),
    )
    write_json(
        directory / "score_summary.json",
        {
            "summary": summary,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_unit": "source_title_cluster",
            "ci_scope": "conditional on the fixed document stream and fitted cumulative state",
            "score_units": "summary: 0-100; per-question scores.csv: 0-1",
            "predictions_sha256": sha256_file(directory / "predictions.jsonl"),
        },
    )
    if plots:
        plot_results(directory / "figures", documents, document_scores, retention)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--plots", action="store_true", help="Write static matplotlib figures"
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be nonnegative")
    print(
        json.dumps(
            score_run(args.run_dir, args.bootstrap_samples, args.plots), indent=2
        )
    )


if __name__ == "__main__":
    main()
