"""Resume saved immediate50 predictions, stopping each seed/method after five limits."""

from .score_stream import qa_scores


def execute(backend, documents, jobs, previous, emit, limit=5):
    docs = {d["doc_id"]: d for d in documents}
    expected = {}
    for job in jobs:
        for doc_id in job["targets"]:
            for qa in docs[doc_id]["questions"]:
                key = (job["group"], job["seed"], job["step"], doc_id, qa["qid"])
                if key in expected:
                    raise ValueError(f"Duplicate planned item: {key}")
                expected[key] = (job, qa)
    old, cache = {}, {}
    fields = ("prediction", "prompt_sha256", "input_tokens", "output_tokens",
              "hit_output_limit", "seconds", "output_token_ids")
    for row in previous:
        key = tuple(row[k] for k in ("group", "seed", "step", "doc_id", "qid"))
        if key in old or key not in expected:
            raise ValueError(f"Duplicate or unexpected previous item: {key}")
        job, qa = expected[key]
        if row["state"] != [list(x) for x in job["state"]] or row["answers"] != qa["answers"]:
            raise ValueError(f"Previous state or answers do not match: {key}")
        if len(row["output_token_ids"]) != row["output_tokens"] or row["hit_output_limit"] != (row["output_tokens"] == 256):
            raise ValueError(f"Invalid previous token metadata: {key}")
        old[key] = row
        state = tuple(map(tuple, job["state"]))
        cache[(state, key[-2], key[-1])] = {k: row[k] for k in fields}
    streak, stopped = {}, {}
    for job in jobs:
        state = tuple(map(tuple, job["state"]))
        method = (job["seed"], job["group"])
        bound = False
        for did in job["targets"]:
            doc = docs[did]
            for qa in doc["questions"]:
                key = (job["group"], job["seed"], job["step"], did, qa["qid"])
                ck = (state, did, qa["qid"])
                if key in old:
                    result = {k: old[key][k] for k in fields}
                    origin = "previous"
                elif job["seed"] is not None and method in stopped:
                    emit("skipped", {"group": job["group"], "seed": job["seed"], "step": job["step"],
                                     "doc_id": did, "qid": qa["qid"], "reason": "five_consecutive_output_limits",
                                     "trigger": stopped[method]})
                    continue
                elif ck in cache:
                    result = cache[ck]
                    origin = "reused"
                else:
                    backend.location = {"group": job["group"], "seed": job["seed"], "step": job["step"],
                                        "doc_id": did, "qid": qa["qid"]}
                    if not bound:
                        backend.bind_weighted(state)
                        bound = True
                    result = backend.answer(doc, qa["question"])
                    cache[ck] = result
                    origin = "new"
                after_trigger = job["seed"] is not None and method in stopped
                emit("predictions", {"group": job["group"], "seed": job["seed"], "step": job["step"],
                    "doc_id": did, "qid": qa["qid"], "state": job["state"], "answers": qa["answers"],
                    "origin": origin, "previous_after_trigger": bool(after_trigger and origin == "previous"),
                    **result, **qa_scores(result["prediction"], qa["answers"])})
                if job["seed"] is not None and method not in stopped:
                    streak[method] = streak.get(method, 0) + 1 if result["hit_output_limit"] else 0
                    if streak[method] == limit:
                        event = {"seed": job["seed"], "group": job["group"], "step": job["step"],
                                 "doc_id": did, "qid": qa["qid"], "consecutive": limit,
                                 "trigger_origin": origin}
                        stopped[method] = event
                        emit("early_stops", event)
    backend.reset()
    if not set(old).issubset(expected):
        raise ValueError("Previous predictions were not accounted for")
    return stopped
