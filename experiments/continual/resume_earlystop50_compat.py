"""Matpool compatibility for resuming the early-stop experiment."""
import runpy
import torch
if not hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals = lambda entries: None
import ctx_to_lora.modeling.idefics2 as idefics2
original = idefics2.unpad_input
def compatible(*args, **kwargs):
    value = original(*args, **kwargs)
    return (*value, None) if len(value) == 4 else value
idefics2.unpad_input = compatible
runpy.run_module("experiments.continual.resume_earlystop50", run_name="__main__")
