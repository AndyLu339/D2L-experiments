import json
import unittest
from experiments.continual.earlystop50 import execute
from experiments.continual.immediate50 import make_jobs, orders_for
from experiments.continual.score_earlystop50 import score


class Backend:
    def reset(self): pass
    def bind_weighted(self, state): pass
    def answer(self, doc, question):
        hit = self.location["group"] == "A1"
        n = 256 if hit else 1
        return {"prediction": "yes", "prompt_sha256": "h", "input_tokens": 1,
                "output_tokens": n, "output_token_ids": [1]*n,
                "hit_output_limit": hit, "seconds": 1}


class EarlyStopTests(unittest.TestCase):
    def test_five_hits_preserves_old_rows_and_scores_skips_as_missing(self):
        docs = [{"doc_id": str(i), "questions": [{"qid": f"q{i}", "question": "?", "answers": ["yes"]}]} for i in range(50)]
        orders = orders_for(docs); jobs = make_jobs(orders)
        old = []
        for j in jobs:
            if j["seed"] == 44 and j["group"] == "A1" and j["step"] <= 7:
                did = j["targets"][0]
                old.append({"group": "A1", "seed": 44, "step": j["step"], "doc_id": did,
                    "qid": f"q{did}", "state": j["state"], "answers": ["yes"],
                    "prediction": "yes", "prompt_sha256": "h", "input_tokens": 1,
                    "output_tokens": 256, "output_token_ids": [1]*256,
                    "hit_output_limit": True, "seconds": 1})
        old = json.loads(json.dumps(old)); events = {"predictions": [], "skipped": [], "early_stops": []}
        execute(Backend(), docs, jobs, old, lambda k,r: events[k].append(json.loads(json.dumps(r))))
        event = next(r for r in events["early_stops"] if r["seed"] == 44 and r["group"] == "A1")
        self.assertEqual(event["step"], 5)
        self.assertEqual(sum(r["origin"] == "previous" for r in events["predictions"]), 7)
        self.assertEqual(sum(r["previous_after_trigger"] for r in events["predictions"]), 2)
        self.assertEqual(sum(r["seed"] == 44 and r["group"] == "A1" for r in events["skipped"]), 43)
        scored, doc_scores = score(docs, events["predictions"], events["skipped"], orders)
        self.assertEqual(len(scored) + len(events["skipped"]), len(jobs))
        skipped_doc = next(r for r in doc_scores if r["seed"] == 44 and r["group"] == "A1" and r["step"] == 8)
        self.assertFalse(skipped_doc["complete"]); self.assertIsNone(skipped_doc["f1"])
        self.assertTrue(any(r["seed"] == 44 and r["group"] == "B2" and r["step"] == 50 for r in scored))


if __name__ == "__main__": unittest.main()
