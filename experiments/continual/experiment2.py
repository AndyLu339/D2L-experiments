"""Experiment 2: fixed documents, weighted LoRA states, and paired evaluations."""

import math
import random

SEEDS = (11, 22, 33)


def orders_for(documents):
    """Seed 11 is the original prefix; other seeds shuffle that exact prefix."""
    ids = [d["doc_id"] for d in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document IDs")
    orders = {11: ids}
    for seed in SEEDS[1:]:
        order = ids.copy()
        random.Random(seed).shuffle(order)
        orders[seed] = order
    return orders


def make_jobs(orders, block_size=5):
    """Return logical evaluations. Identical ordered states can reuse predictions."""
    reference = orders[11]
    if not reference or len(reference) % block_size:
        raise ValueError("Document count must be a positive multiple of block size")
    if set(orders) != set(SEEDS) or any(
        len(order) != len(reference) or set(order) != set(reference)
        or len(set(order)) != len(order) for order in orders.values()
    ):
        raise ValueError("Every seed must contain the same unique documents")
    jobs = []

    def add(group, seed, step, weights, targets, first=None):
        # Omit zero contributions: gamma=0 is exactly the single-doc state.
        state = [(doc, float(c)) for doc, c in weights if c != 0]
        jobs.append(dict(group=group, seed=seed, step=step,
                         block=(step // block_size if seed is not None else None),
                         state=state, targets=list(targets), first=first))

    for doc in reference:
        add("base", None, 0, [], [doc])
        for group, coefficient in (("reset_1", 1), ("reset_0.5", .5), ("reset_0.25", .25)):
            add(group, None, 0, [(doc, coefficient)], [doc])

    for seed, order in orders.items():
        for step in range(block_size, len(order) + 1, block_size):
            seen, current = order[:step], order[step-block_size:step]
            first = current[0]
            add("A1", seed, step, [(d, 1) for d in seen], current, first)
            add("A2", seed, step,
                [(d, 2 if i % block_size == 0 else 1) for i, d in enumerate(seen)],
                current, first)
            add("A3", seed, step, [(d, 1) for d in current], current, first)
            for group, coefficient in (("B2", .5), ("B3", .25),
                                       ("B4", 1 / math.sqrt(step)), ("B5", 1 / step)):
                add(group, seed, step, [(d, coefficient) for d in seen], current, first)
            for group, gamma in (("C0", 0), ("C0.25", .25), ("C0.5", .5), ("C1", 1)):
                add(group, seed, step,
                    [(d, 1 if d == first else gamma) for d in seen], [first], first)
    return jobs


def pilot_jobs(jobs, order):
    """Measure first seed's first five documents, without calling it a full run."""
    first_five = set(order[:5])
    return [job for job in jobs if
            (job["seed"] is None and job["targets"][0] in first_five) or
            (job["seed"] == 11 and job["step"] == 5)]


def run_jobs(backend, documents, jobs, emit):
    """Cache predictions only for identical ordered, nonzero states and questions."""
    from .score_stream import qa_scores

    by_id = {doc["doc_id"]: doc for doc in documents}
    cache = {}
    for job in jobs:
        state = tuple(tuple(pair) for pair in job["state"])
        targets = [by_id[d] for d in job["targets"]]
        needed = [(doc, qa) for doc in targets for qa in doc["questions"]
                  if (state, doc["doc_id"], qa["qid"]) not in cache]
        new_keys = {(state, doc["doc_id"], qa["qid"]) for doc, qa in needed}
        if needed:
            backend.location = {k: job[k] for k in ("group", "seed", "step")}
            backend.bind_weighted(state)
            for doc, qa in needed:
                backend.location = {**backend.location, "doc_id": doc["doc_id"], "qid": qa["qid"]}
                cache[(state, doc["doc_id"], qa["qid"])] = backend.answer(doc, qa["question"])
        for doc in targets:
            for qa in doc["questions"]:
                result = cache[(state, doc["doc_id"], qa["qid"])]
                emit("predictions", {**{k: job[k] for k in ("group", "seed", "step", "block")},
                    "doc_id": doc["doc_id"], "qid": qa["qid"],
                    "prediction_reused": (state, doc["doc_id"], qa["qid"]) not in new_keys,
                    "is_first": job["first"] == doc["doc_id"],
                    "state": job["state"], "answers": qa["answers"],
                    **result, **qa_scores(result["prediction"], qa["answers"])})
    backend.reset()
