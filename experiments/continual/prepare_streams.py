"""Group the FULL SQuAD validation split before sampling documents."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from .common import sha256_file, write_json


def read_local_squad(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            yield from (json.loads(line) for line in handle if line.strip())
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        yield from payload
        return
    if payload.get("version") not in (None, "1.1"):
        raise ValueError("Only SQuAD 1.1 is supported")
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if qa.get("is_impossible", False):
                    raise ValueError("SQuAD 2.0 unanswerable questions are unsupported")
                yield {
                    "id": qa["id"],
                    "title": article["title"],
                    "context": paragraph["context"],
                    "question": qa["question"],
                    "answers": [answer["text"] for answer in qa["answers"]],
                }


def group_documents(rows):
    groups, source_ids = {}, {}
    row_count = 0
    for row in rows:
        row_count += 1
        context, question = row["context"], row["question"]
        answers = row["answers"]
        if isinstance(answers, dict):
            answers = answers["text"]
        if not context.strip() or not question.strip() or not answers:
            raise ValueError("Each SQuAD question needs context, question, and answers")
        if not all(isinstance(answer, str) and answer.strip() for answer in answers):
            raise ValueError("Reference answers must be nonempty strings")
        doc_id = hashlib.sha256(context.encode()).hexdigest()
        qid = str(row["id"])
        identity = (doc_id, question)
        if qid in source_ids and source_ids[qid] != identity:
            raise ValueError(f"Conflicting source question ID: {qid}")
        source_ids[qid] = identity
        doc = groups.setdefault(
            doc_id, {"doc_id": doc_id, "context": context, "titles": set(), "qas": {}}
        )
        doc["titles"].add(row.get("title", ""))
        qa = doc["qas"].setdefault(
            question, {"question": question, "source_ids": set(), "answers": set()}
        )
        qa["source_ids"].add(qid)
        qa["answers"].update(answers)
    documents = []
    for doc_id, doc in sorted(groups.items()):
        questions = []
        for qa in doc.pop("qas").values():
            ids = sorted(qa["source_ids"])
            questions.append(
                {
                    **qa,
                    "qid": ids[0],
                    "source_ids": ids,
                    "answers": sorted(qa["answers"]),
                }
            )
        documents.append(
            {
                **doc,
                "titles": sorted(doc["titles"]),
                "questions": sorted(questions, key=lambda qa: qa["qid"]),
            }
        )
    return documents, row_count


def prepare(rows, output_dir, num_docs, sample_seed, order_seeds, source):
    documents, row_count = group_documents(rows)
    if not 1 <= num_docs <= len(documents):
        raise ValueError(
            f"Requested {num_docs} documents; only {len(documents)} available"
        )
    selected = random.Random(sample_seed).sample(documents, num_docs)
    selected.sort(key=lambda doc: doc["doc_id"])
    streams = {}
    for seed in order_seeds:
        order = [doc["doc_id"] for doc in selected]
        random.Random(seed).shuffle(order)
        streams[str(seed)] = order
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    document_path = output_dir / "documents.jsonl"
    with document_path.open("w", encoding="utf-8") as handle:
        for doc in selected:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "dataset": "squad_v1.1",
        "split": "validation",
        "source": source,
        "source_row_count": row_count,
        "source_document_count": len(documents),
        "selected_document_count": len(selected),
        "selected_question_count": sum(len(doc["questions"]) for doc in selected),
        "sample_seed": sample_seed,
        "documents_sha256": sha256_file(document_path),
        "streams": streams,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, help="Official dev-v1.1.json or flat JSONL"
    )
    parser.add_argument("--dataset-source", default="rajpurkar/squad")
    parser.add_argument("--revision", help="HF dataset revision (prefer a commit SHA)")
    parser.add_argument("--num-docs", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--order-seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.input:
        if args.revision:
            parser.error("--revision applies only to --dataset-source")
        rows = read_local_squad(args.input)
        source = {"path": str(args.input.resolve()), "sha256": sha256_file(args.input)}
    else:
        from datasets import load_dataset

        rows = load_dataset(
            args.dataset_source, split="validation", revision=args.revision
        )
        source = {
            "path": args.dataset_source,
            "revision": args.revision,
            "fingerprint": rows._fingerprint,
        }
    manifest = prepare(
        rows, args.output_dir, args.num_docs, args.sample_seed, args.order_seeds, source
    )
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "streams"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
