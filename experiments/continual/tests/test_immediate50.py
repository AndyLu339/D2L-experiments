import json
import math
import unittest

from experiments.continual.experiment2 import run_jobs
from experiments.continual.immediate50 import make_jobs, orders_for
from experiments.continual.score_immediate50 import score


class Backend:
    def __init__(self):
        self.calls = 0
        self.state = None

    def bind_weighted(self, state):
        self.state = state

    def answer(self, doc, question):
        self.calls += 1
        return {"prediction": "yes", "output_token_ids": [1], "output_tokens": 1,
                "hit_output_limit": False, "seconds": 0}

    def reset(self):
        self.state = None


class Immediate50Tests(unittest.TestCase):
    def setUp(self):
        self.docs = [{"doc_id": str(i), "questions": [
            {"qid": f"q{i}", "question": "?", "answers": ["yes"]}]} for i in range(50)]
        self.orders = orders_for(self.docs)
        self.jobs = make_jobs(self.orders)

    def test_orders_baselines_and_step_specific_coefficients(self):
        self.assertEqual(set(self.orders), {44, 55, 66, 77})
        self.assertEqual(len({tuple(order) for order in self.orders.values()}), 4)
        self.assertTrue(all(set(order) == set(self.orders[44]) for order in self.orders.values()))
        self.assertEqual(len(self.jobs), 50 * 3 + 4 * 50 * 5)
        for group, expected in (("A1", 1), ("B2", .5), ("B3", .25),
                                ("B4", 1 / math.sqrt(20)), ("B5", 1 / 20)):
            job = next(j for j in self.jobs if (j["group"], j["seed"], j["step"]) == (group, 44, 20))
            self.assertEqual(job["targets"], [self.orders[44][19]])
            self.assertEqual([doc for doc, _ in job["state"]], self.orders[44][:20])
            self.assertTrue(all(coef == expected for _, coef in job["state"]))
        self.assertFalse(any(j["group"] in ("A2", "A3") for j in self.jobs))

    def test_full_protocol_reuse_and_scoring_validation(self):
        backend, rows = Backend(), []
        run_jobs(backend, self.docs, self.jobs,
                 lambda kind, row: rows.append(json.loads(json.dumps(row))))
        self.assertEqual(len(rows), 50 * 3 + 4 * 50 * 5)
        self.assertLess(backend.calls, len(rows))
        self.assertIsNone(backend.state)
        scored, document_scores = score(self.docs, rows, self.orders)
        self.assertEqual(len(scored), len(rows))
        self.assertEqual(len(document_scores), len(rows))
        self.assertTrue(all(row["f1"] == 100 for row in document_scores))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            score(self.docs, rows[:-1], self.orders)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            score(self.docs, rows + rows[:1], self.orders)
        changed = json.loads(json.dumps(rows))
        changed[-1]["state"][0][1] += 1
        with self.assertRaisesRegex(ValueError, "State"):
            score(self.docs, changed, self.orders)


if __name__ == "__main__":
    unittest.main()
