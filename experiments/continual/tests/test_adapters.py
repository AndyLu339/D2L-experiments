from functools import partial
from types import SimpleNamespace
import unittest

try:
    import torch
    from torch import nn
    from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers, lora_forward
    from ctx_to_lora.modeling.lora_merger import combine_lora
except ImportError as exc:
    raise unittest.SkipTest(f"D2L tensor test dependencies unavailable: {exc}")

from experiments.continual.adapters import CumulativeLoRA, adapter_fingerprint


class ToyModel:
    """Small CPU linears using the repository's actual merger and forward hooks."""

    def __init__(self, use_bias=True):
        self.base_model = nn.Module()
        self.base_model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        self.originals = []
        for layer in self.base_model.layers:
            layer.mlp = nn.Module()
            layer.mlp.down_proj = nn.Linear(4, 3, bias=False)
            self.originals.append(layer.mlp.down_proj.forward)
        bias = {"down_proj": {"A": torch.randn(2, 2, 4), "B": torch.randn(2, 2, 3)}}
        self.hypernet = SimpleNamespace(
            layer_indices=range(2),
            config=SimpleNamespace(use_bias=use_bias),
            get_head_bias=lambda: bias,
        )
        self.generated_loras = None

    combine_lora = staticmethod(combine_lora)
    apply_lora_to_layers = staticmethod(apply_lora_to_layers)

    def reset(self):
        for layer, original in zip(self.base_model.layers, self.originals):
            layer.mlp.down_proj.forward = original

    def patch_lora_forward(self):
        for layer in self.base_model.layers:
            layer.mlp.down_proj.forward = partial(
                lora_forward, self=layer.mlp.down_proj, lora_dropout_p=0, scaling=0.7
            )


def delta(chunks=1):
    return {
        "down_proj": {
            "A": torch.randn(chunks, 2, 2, 4),
            "B": torch.randn(chunks, 2, 2, 3),
        }
    }


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)

    def test_cumulative_forward_is_sum_plus_shared_bias_once(self):
        model = ToyModel()
        state = CumulativeLoRA(model)
        first, second = delta(), delta(2)
        first_hash = adapter_fingerprint(first)
        x = torch.randn(1, 5, 4)
        baselines = [fn(x) for fn in model.originals]
        state.append(first)
        state.bind()
        # A second bind and many queries must not compound the applied update.
        state.bind()
        state.append(second)
        state.bind()
        self.assertEqual(state.info()["num_chunks"], 3)
        self.assertEqual(state.info()["physical_rank"], 8)
        self.assertEqual(first_hash, adapter_fingerprint(first))
        before = adapter_fingerprint(state.raw)
        for index, layer in enumerate(model.base_model.layers):
            matrix = torch.zeros(3, 4)
            for part in (first, second):
                pair = part["down_proj"]
                matrix += sum(
                    b.T @ a for a, b in zip(pair["A"][:, index], pair["B"][:, index])
                )
            bias = model.hypernet.get_head_bias()["down_proj"]
            matrix += bias["B"][index].T @ bias["A"][index]
            expected = baselines[index] + 0.7 * x @ matrix.T
            for _ in range(3):
                torch.testing.assert_close(layer.mlp.down_proj(x), expected)
        self.assertEqual(before, adapter_fingerprint(state.raw))
        state.clear()
        for index, layer in enumerate(model.base_model.layers):
            torch.testing.assert_close(layer.mlp.down_proj(x), baselines[index])

    def test_single_document_matches_existing_combiner(self):
        model, raw = ToyModel(), delta(2)
        state = CumulativeLoRA(model)
        state.append(raw)
        state.bind()
        x = torch.randn(1, 2, 4)
        result = model.base_model.layers[0].mlp.down_proj(x)
        model.reset()
        model.patch_lora_forward()
        combined = combine_lora(raw, torch.tensor([2]), model.hypernet.get_head_bias())
        apply_lora_to_layers(model.base_model, range(2), combined, torch.tensor([1]))
        torch.testing.assert_close(result, model.base_model.layers[0].mlp.down_proj(x))

    def test_final_state_commutes_without_repeating_bias(self):
        model = ToyModel(use_bias=False)
        first, second = delta(), delta(2)
        x = torch.randn(1, 1, 4)
        state = CumulativeLoRA(model)
        outputs = []
        for parts in ((first, second), (second, first)):
            state.clear()
            for part in parts:
                state.append(part)
            state.bind()
            outputs.append(model.base_model.layers[0].mlp.down_proj(x))
        torch.testing.assert_close(*outputs)

    def test_incompatible_or_nonfinite_delta_is_rejected(self):
        state = CumulativeLoRA(ToyModel())
        state.append(delta())
        invalid = delta()
        invalid["down_proj"]["A"] = torch.zeros(1, 2, 3, 4)
        with self.assertRaises(ValueError):
            state.append(invalid)
        invalid = delta()
        invalid["down_proj"]["A"][0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            state.append(invalid)


if __name__ == "__main__":
    unittest.main()
