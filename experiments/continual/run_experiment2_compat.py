"""Run experiment 2 in the existing Matpool Torch 2.1 / FlashAttention 2.6 environment."""

import runpy

import torch

if not hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals = lambda entries: None

import ctx_to_lora.modeling.idefics2 as idefics2

_original_unpad_input = idefics2.unpad_input


def _compatible_unpad_input(*args, **kwargs):
    result = _original_unpad_input(*args, **kwargs)
    return (*result, None) if len(result) == 4 else result


idefics2.unpad_input = _compatible_unpad_input
runpy.run_module("experiments.continual.run_experiment2", run_name="__main__")
