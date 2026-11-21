import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_stream(directory, order_seed, num_docs=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    document_path = directory / "documents.jsonl"
    if sha256_file(document_path) != manifest["documents_sha256"]:
        raise ValueError("documents.jsonl has changed since manifest creation")
    documents = read_jsonl(document_path)
    by_id = {doc["doc_id"]: doc for doc in documents}
    if len(by_id) != len(documents):
        raise ValueError("Duplicate document IDs")
    for doc in documents:
        if hashlib.sha256(doc["context"].encode()).hexdigest() != doc["doc_id"]:
            raise ValueError("Document ID does not match context hash")
    order = manifest["streams"][str(order_seed)]
    if len(order) != len(by_id) or set(order) != set(by_id):
        raise ValueError("Stream must contain each selected document exactly once")
    if num_docs is not None and not 1 <= num_docs <= len(order):
        raise ValueError(f"num_docs must be between 1 and {len(order)}")
    return manifest, [by_id[doc_id] for doc_id in order[:num_docs]]
