import tempfile
import unittest
from pathlib import Path

from experiments.continual.common import load_stream
from experiments.continual.prepare_streams import group_documents, prepare


def rows():
    return [
        {
            "id": "a1",
            "title": "A",
            "context": "A red fox.",
            "question": "Color?",
            "answers": {"text": ["red", "Red"]},
        },
        {
            "id": "b1",
            "title": "B",
            "context": "A blue bird.",
            "question": "Color?",
            "answers": {"text": ["blue"]},
        },
        {
            "id": "a2",
            "title": "A",
            "context": "A red fox.",
            "question": "Animal?",
            "answers": {"text": ["fox"]},
        },
        {
            "id": "a3",
            "title": "A",
            "context": "A red fox.",
            "question": "Color?",
            "answers": {"text": ["red"]},
        },
    ]


class PrepareTests(unittest.TestCase):
    def test_groups_nonadjacent_context_and_retains_all_answers(self):
        documents, count = group_documents(rows())
        self.assertEqual(count, 4)
        fox = next(doc for doc in documents if "fox" in doc["context"])
        self.assertEqual(len(fox["questions"]), 2)
        color = next(q for q in fox["questions"] if q["question"] == "Color?")
        self.assertEqual(color["answers"], ["Red", "red"])
        self.assertEqual(color["source_ids"], ["a1", "a3"])

    def test_reproducible_selection_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            one, two = Path(temp) / "one", Path(temp) / "two"
            a = prepare(rows(), one, 2, 42, [11, 22], {})
            b = prepare(reversed(rows()), two, 2, 42, [11, 22], {})
            self.assertEqual(a, b)
            _, stream = load_stream(one, 11, 1)
            self.assertEqual(len(stream), 1)
            with (one / "documents.jsonl").open("a") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_stream(one, 11)

    def test_conflicting_ids_are_rejected(self):
        data = rows()
        data[1]["id"] = "a1"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            group_documents(data)


if __name__ == "__main__":
    unittest.main()
