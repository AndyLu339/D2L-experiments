"""Single-document-at-a-time evaluation for fixed 50-document orders."""

import math
import random

SEEDS = (44, 55, 66, 77)


def orders_for(documents):
    ids = [d["doc_id"] for d in documents]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError("The experiment requires exactly 50 distinct documents")
    result = {}
    for seed in SEEDS:
        order = ids.copy()
        random.Random(seed).shuffle(order)
        result[seed] = order
    return result


def make_jobs(orders):
    if set(orders) != set(SEEDS):
        raise ValueError("Expected four configured seeds")
    ids = orders[SEEDS[0]]
    if len(ids) != 50 or len(set(ids)) != 50 or any(
        len(order) != 50 or set(order) != set(ids) for order in orders.values()
    ):
        raise ValueError("Orders must contain the same 50 unique documents")
    jobs = []

    def add(group, seed, step, state, doc):
        jobs.append({"group": group, "seed": seed, "step": step, "block": None,
                     "state": [(d, float(c)) for d, c in state],
                     "targets": [doc], "first": None})

    # Each independent baseline is evaluated once per document and reused for all seeds.
    for doc in ids:
        for group, coef in (("reset_1", 1), ("reset_0.5", .5), ("reset_0.25", .25)):
            add(group, None, 0, [(doc, coef)], doc)
    for seed, order in orders.items():
        for step, doc in enumerate(order, 1):
            seen = order[:step]
            for group, coefficient in (("A1", 1), ("B2", .5), ("B3", .25),
                                       ("B4", 1 / math.sqrt(step)), ("B5", 1 / step)):
                add(group, seed, step, [(d, coefficient) for d in seen], doc)
    return jobs
