import json
from pathlib import Path
import tempfile
import unittest

from experiments.continual.common import sha256_file, write_json
from experiments.continual.protocol import run_protocol
from experiments.continual.score_stream import (
    paired_interval,
    qa_scores,
    score_predictions,
    score_run,
    summarize,
)


def example_documents(count=3):
    return [
        {
            "doc_id": str(i),
            "context": f"Document {i}",
            "titles": [f"Title {i}"],
            "questions": [
                {"qid": str(i), "question": "Which token?", "answers": ["alpha"]}
            ],
        }
        for i in range(count)
    ]


class ScoringTests(unittest.TestCase):
    def test_official_normalization_and_multiple_references(self):
        self.assertEqual(qa_scores("The RED!", ["blue", "red"]), {"em": 1, "f1": 1})
        self.assertEqual(qa_scores("red-blue", ["red blue"]), {"em": 0, "f1": 0})
        self.assertAlmostEqual(qa_scores("red fox", ["fox"])["f1"], 2 / 3)
        # SQuAD 1.1 assigns zero F1 when there are no overlapping tokens, even
        # when both normalized strings are empty (different from v2 handling).
        self.assertEqual(qa_scores("a", ["the"]), {"em": 1, "f1": 0})

    def test_five_columns_match_three_document_example(self):
        docs = example_documents()
        scores = {}
        for i, doc in enumerate(docs, 1):
            for group, phase, step, value in (
                ("reset", "independent", i, 0.8),
                ("seq", "immediate", i, 0.8),
                ("seq", "retention", 3, (0.6, 0.7, 0.8)[i - 1]),
            ):
                scores[(group, phase, step, doc["doc_id"])] = {
                    "em": value,
                    "f1": value,
                    "question_count": 1,
                }
        rows, _ = summarize(docs, scores, ["reset", "seq"], [3], bootstrap_samples=50)
        for row in rows:
            self.assertAlmostEqual(row["seq_online"], 80)
            self.assertAlmostEqual(row["seq_final"], 70)
            self.assertAlmostEqual(row["immediate_gap"], 0)
            self.assertAlmostEqual(row["final_gap"], -10)
            self.assertAlmostEqual(row["bwt"], -15)
            self.assertAlmostEqual(row["checkpoint_forgetting"], 15)

    def test_macro_and_micro_have_different_weights(self):
        docs = example_documents(2)
        docs[1]["questions"] *= 3
        scores = {
            ("reset", "independent", i, doc["doc_id"]): {
                "em": float(i == 1),
                "f1": float(i == 1),
            }
            for i, doc in enumerate(docs, 1)
        }
        rows, _ = summarize(docs, scores, ["reset"], [], bootstrap_samples=0)
        self.assertEqual(rows[0]["reset"], 50)
        self.assertEqual(rows[1]["reset"], 25)
        self.assertIsNone(rows[0]["seq_online"])

    def test_missing_or_duplicate_predictions_are_errors(self):
        docs = example_documents(1)
        row = {
            "group": "reset",
            "phase": "independent",
            "step": 1,
            "doc_id": "0",
            "qid": "0",
            "prediction": "alpha",
        }
        with self.assertRaisesRegex(ValueError, "missing"):
            score_predictions(docs, [], ["reset"], [])
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            score_predictions(docs, [row, row], ["reset"], [])

    def test_bootstrap_keeps_constant_paired_difference(self):
        low, high = paired_interval(example_documents(), [-0.1] * 3, [1, 2, 3], 100, 42)
        self.assertAlmostEqual(low, -0.1)
        self.assertAlmostEqual(high, -0.1)

    def test_protocol_outputs_score_end_to_end_and_failed_runs_are_rejected(self):
        class Backend:
            def reset(self):
                self.memory = []

            def encode(self, doc):
                return doc["doc_id"]

            def update(self, delta):
                self.memory.append(delta)

            def state_info(self):
                return {"num_documents": len(self.memory)}

            def answer(self, doc, question, include_context=False):
                correct = include_context or (
                    self.memory and self.memory[-1] == doc["doc_id"]
                )
                return {"prediction": "alpha" if correct else "wrong"}

        docs, groups, checkpoints = (
            example_documents(),
            ["base", "icl", "reset", "seq"],
            [1, 3],
        )
        events = []
        run_protocol(
            Backend(),
            docs,
            checkpoints,
            groups,
            lambda name, row: events.append((name, row)),
        )
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            snapshot = directory / "evaluation_documents.jsonl"
            snapshot.write_text("".join(json.dumps(doc) + "\n" for doc in docs))
            (directory / "predictions.jsonl").write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for name, row in events
                    if name == "predictions"
                )
            )
            (directory / "resources.jsonl").write_text("")
            write_json(
                directory / "config.json",
                {
                    "evaluation_documents_sha256": sha256_file(snapshot),
                    "doc_ids": [doc["doc_id"] for doc in docs],
                    "groups": groups,
                    "checkpoints": checkpoints,
                },
            )
            write_json(directory / "status.json", {"status": "completed"})
            rows = score_run(directory, bootstrap_samples=10)
            self.assertEqual(rows[0]["seq_online"], 100)
            self.assertAlmostEqual(rows[0]["seq_final"], 100 / 3)
            self.assertEqual(rows[0]["bwt"], -100)
            self.assertTrue((directory / "summary.csv").exists())
            write_json(directory / "status.json", {"status": "failed"})
            with self.assertRaisesRegex(ValueError, "not completed"):
                score_run(directory)


if __name__ == "__main__":
    unittest.main()
