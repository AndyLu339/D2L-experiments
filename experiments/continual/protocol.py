"""Evaluation order, independent of model loading or scoring."""


def run_protocol(backend, documents, checkpoints, groups, emit):
    def evaluate(doc, group, phase, step):
        for qa in doc["questions"]:
            backend.location = {
                "group": group,
                "phase": phase,
                "step": step,
                "doc_id": doc["doc_id"],
                "qid": qa["qid"],
            }
            result = backend.answer(
                doc, qa["question"], include_context=(group == "icl")
            )
            emit(
                "predictions",
                {
                    "group": group,
                    "phase": phase,
                    "step": step,
                    "doc_id": doc["doc_id"],
                    "qid": qa["qid"],
                    **result,
                },
            )

    for group in ("base", "icl", "reset"):
        if group not in groups:
            continue
        for step, doc in enumerate(documents, 1):
            backend.location = {"group": group, "phase": "independent", "step": step}
            backend.reset()
            if group == "reset":
                delta = backend.encode(doc)
                backend.update(delta)
                del delta
                emit(
                    "states",
                    {
                        **backend.location,
                        "doc_id": doc["doc_id"],
                        **backend.state_info(),
                    },
                )
            evaluate(doc, group, "independent", step)

    if "seq" in groups:
        backend.reset()
        for step, doc in enumerate(documents, 1):
            backend.location = {"group": "seq", "phase": "immediate", "step": step}
            delta = backend.encode(doc)
            backend.update(delta)
            del delta
            emit(
                "states",
                {**backend.location, "doc_id": doc["doc_id"], **backend.state_info()},
            )
            evaluate(doc, "seq", "immediate", step)
            if step in checkpoints:
                backend.location = {"group": "seq", "phase": "retention", "step": step}
                for old_doc in documents[:step]:
                    evaluate(old_doc, "seq", "retention", step)
    backend.reset()
