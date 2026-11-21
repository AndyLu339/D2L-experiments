import unittest

from experiments.continual.protocol import run_protocol


class RecordingBackend:
    def __init__(self):
        self.memory, self.encodes, self.calls = [], [], []

    def reset(self):
        self.memory = []

    def encode(self, doc):
        self.encodes.append((self.location["group"], doc["doc_id"]))
        return doc["doc_id"]

    def update(self, delta):
        self.memory.append(delta)

    def state_info(self):
        return {"num_documents": len(self.memory)}

    def answer(self, doc, question, include_context=False):
        self.calls.append((self.location.copy(), include_context, self.memory.copy()))
        return {"prediction": ",".join(self.memory)}


class ProtocolTests(unittest.TestCase):
    def test_online_final_and_reset_states_are_distinct(self):
        docs = [
            {
                "doc_id": str(i),
                "questions": [
                    {"qid": f"{i}-{j}", "question": f"Question {j}"} for j in range(2)
                ],
            }
            for i in range(3)
        ]
        backend, events = RecordingBackend(), []
        run_protocol(
            backend,
            docs,
            [1, 3],
            ["base", "icl", "reset", "seq"],
            lambda kind, row: events.append((kind, row)),
        )
        predictions = [row for kind, row in events if kind == "predictions"]
        self.assertEqual(len(predictions), 32)
        for row in predictions:
            if row["group"] == "reset":
                self.assertEqual(row["prediction"], row["doc_id"])
            if row["group"] == "seq":
                self.assertEqual(
                    row["prediction"], ",".join(map(str, range(row["step"])))
                )
        for group in ("reset", "seq"):
            self.assertEqual(
                [doc for g, doc in backend.encodes if g == group], ["0", "1", "2"]
            )
        for location, include_context, memory in backend.calls:
            self.assertEqual(include_context, location["group"] == "icl")
            if location["group"] in ("base", "icl"):
                self.assertEqual(memory, [])
        self.assertEqual(backend.memory, [])


if __name__ == "__main__":
    unittest.main()
