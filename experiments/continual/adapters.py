"""Accumulate raw chunk adapters; reuse the author's merger and layer hooks."""

import hashlib

import torch


def adapter_fingerprint(raw):
    digest = hashlib.sha256()
    for module in sorted(raw):
        for name in ("A", "B"):
            tensor = raw[module][name].detach().cpu().contiguous()
            digest.update(f"{module}/{name}/{tensor.shape}/{tensor.dtype}".encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class CumulativeLoRA:
    def __init__(self, model):
        self.model = model
        self.raw = None
        self.num_documents = 0
        self.chain_hash = ""
        self.bound_rank = 0

    def clear(self):
        self.model.reset()
        self.raw = None
        self.num_documents = 0
        self.chain_hash = ""
        self.bound_rank = 0

    @torch.inference_mode()
    def append(self, delta):
        if not delta:
            raise ValueError("No generated LoRA modules")
        chunk_counts = set()
        for pair in delta.values():
            for name in ("A", "B"):
                value = pair[name]
                if value.ndim != 4 or not all(value.shape):
                    raise ValueError("Expected nonempty [chunks, layers, rank, dim]")
                if not torch.isfinite(value).all():
                    raise ValueError("Non-finite LoRA weights")
                chunk_counts.add(value.shape[0])
        if len(chunk_counts) != 1:
            raise ValueError("LoRA modules disagree on number of chunks")
        if self.raw is not None:
            if self.raw.keys() != delta.keys():
                raise ValueError("LoRA module names changed")
            for module, pair in delta.items():
                for name in ("A", "B"):
                    old, new = self.raw[module][name], pair[name]
                    if (
                        old.shape[1:] != new.shape[1:]
                        or old.dtype != new.dtype
                        or old.device != new.device
                    ):
                        raise ValueError("LoRA shape, precision, or device changed")
        delta_hash = adapter_fingerprint(delta)
        if self.raw is None:
            self.raw = {
                m: {k: p[k].detach() for k in ("A", "B")} for m, p in delta.items()
            }
        else:
            self.raw = {
                m: {k: torch.cat((self.raw[m][k], p[k]), dim=0) for k in ("A", "B")}
                for m, p in delta.items()
            }
        self.num_documents += 1
        self.chain_hash = hashlib.sha256(
            (self.chain_hash + delta_hash).encode()
        ).hexdigest()
        return delta_hash

    @torch.inference_mode()
    def bind(self):
        if self.raw is None:
            raise ValueError("Cannot bind an empty state")
        # Release old forward closures before constructing the next combined state.
        self.model.reset()
        self.model.patch_lora_forward()
        sample = next(iter(self.raw.values()))["A"]
        chunks = torch.tensor([sample.shape[0]], device=sample.device)
        combined = self.model.combine_lora(
            self.raw,
            chunks,
            lora_bias=self.model.hypernet.get_head_bias()
            if self.model.hypernet.config.use_bias
            else None,
        )
        self.model.apply_lora_to_layers(
            self.model.base_model,
            self.model.hypernet.layer_indices,
            combined,
            torch.ones(1, dtype=torch.int32, device=sample.device),
        )
        self.bound_rank = next(iter(combined.values()))["A"].shape[-2]
        # Do not set model.generated_loras: answers use base_model.generate() and
        # the bound hooks directly, so no per-question recombination occurs.

    def info(self):
        if self.raw is None:
            return {
                "num_documents": 0,
                "num_chunks": 0,
                "physical_rank": 0,
                "raw_adapter_bytes": 0,
                "chain_hash": "",
            }
        sample = next(iter(self.raw.values()))["A"]
        return {
            "num_documents": self.num_documents,
            "num_chunks": sample.shape[0],
            "base_rank": sample.shape[-2],
            "physical_rank": self.bound_rank,
            "dtype": str(sample.dtype),
            "raw_adapter_bytes": sum(
                t.numel() * t.element_size()
                for pair in self.raw.values()
                for t in pair.values()
            ),
            "chain_hash": self.chain_hash,
        }
