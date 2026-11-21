import json
import unittest

from experiments.continual.experiment2 import make_jobs, orders_for, pilot_jobs, run_jobs
from experiments.continual.score_experiment2 import summarize


def documents():
    return [{"doc_id": str(i), "context": str(i),
             "questions": [{"qid": f"q{i}", "question": "?", "answers": ["yes"]}]}
            for i in range(20)]


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


class Experiment2Tests(unittest.TestCase):
    def setUp(self):
        self.docs = documents()
        self.orders = orders_for(self.docs)
        self.jobs = make_jobs(self.orders)

    def job(self, group, seed, step):
        return next(j for j in self.jobs if (j["group"], j["seed"], j["step"]) == (group, seed, step))

    def test_orders_and_five_document_states(self):
        self.assertEqual(self.orders[11], [str(i) for i in range(20)])
        for order in self.orders.values():
            self.assertEqual(set(order), set(self.orders[11]))
        self.assertNotEqual(self.orders[11], self.orders[22])
        normal = self.job("A1", 11, 10)
        self.assertEqual(normal["state"], [(str(i), 1.0) for i in range(10)])
        boosted = self.job("A2", 11, 10)
        self.assertEqual([w for _, w in boosted["state"]], [2,1,1,1,1,2,1,1,1,1])
        reset = self.job("A3", 11, 10)
        self.assertEqual(reset["state"], [(str(i), 1.0) for i in range(5,10)])
        self.assertEqual(reset["targets"], [str(i) for i in range(5,10)])
        self.assertEqual(self.job("C0", 11, 10)["state"], [("5", 1.0)])
        self.assertEqual(dict(self.job("C0.25", 11, 10)["state"])["5"], 1)
        self.assertEqual(dict(self.job("C0.25", 11, 10)["state"])["4"], .25)
        self.assertEqual(dict(self.job("B5", 11, 10)["state"])["0"], .1)

    def test_full_protocol_scoring_and_missing_prediction_rejected(self):
        backend, rows = Backend(), []
        run_jobs(backend, self.docs, self.jobs,
                 lambda kind, row: rows.append(json.loads(json.dumps(row))))
        self.assertLess(backend.calls, len(rows))
        self.assertIsNone(backend.state)
        scored, summaries, paired = summarize(self.docs, self.orders, rows)
        self.assertEqual(len(scored), len(rows))
        self.assertTrue(all(row["f1_micro"] == 100 for row in summaries))
        self.assertTrue(all(row["f1_diff"] == 0 for row in paired))
        self.assertEqual(len(paired), 3 * (20 * 10 + 4 * 4))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize(self.docs, self.orders, rows[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize(self.docs, self.orders, rows + rows[:1])

    def test_pilot_uses_only_first_five_documents_and_first_checkpoint(self):
        jobs = pilot_jobs(self.jobs, self.orders[11])
        for job in jobs:
            self.assertTrue(set(job["targets"]).issubset(set(self.orders[11][:5])))
            self.assertTrue(all(doc in self.orders[11][:5] for doc, _ in job["state"]))
            self.assertTrue(job["seed"] is None or (job["seed"] == 11 and job["step"] == 5))


if __name__ == "__main__":
    unittest.main()
