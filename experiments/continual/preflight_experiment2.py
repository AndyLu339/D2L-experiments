"""Real-model checks before experiment 2; not a replacement for evaluation."""

def preflight(backend, documents):
    import torch
    import torch.nn.functional as F
    from ctx_to_lora.utils import get_layers
    from .adapters import adapter_fingerprint

    model = backend.model
    first, second = documents[:2]
    backend.reset()
    chunks, metadata = backend.tokenize_document(first)
    if len(chunks) != 1:
        raise ValueError("Preflight expects single-chunk documents")
    ctx = torch.tensor([chunks[0]], device=model.device)
    model._internalize_from_ids(ctx, torch.ones_like(ctx))
    author_raw = model.generated_loras
    backend.location = {"group": "preflight", "step": 1}
    runner_raw = backend.encode(first)
    for module in runner_raw:
        for key in ("A", "B"):
            torch.testing.assert_close(author_raw[module][key], runner_raw[module][key], rtol=1e-4, atol=1e-5)

    question = first["questions"][0]["question"]
    prompt = backend.prompt_template.format(input=question)
    ids = backend.tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}, {"role": "user", "content": prompt}],
        add_generation_prompt=True, add_special_tokens=False, return_tensors="pt").to(model.device)
    mask = torch.ones_like(ids)
    author_tokens = model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                                   max_new_tokens=backend.args.max_new_tokens, use_cache=True)[0, ids.shape[-1]:]
    author_logits = model.base_model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
    backend.reset()
    base_logits = model.base_model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
    backend.adapters.append(runner_raw)
    backend.adapters.bind()
    runner_logits = model.base_model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
    torch.testing.assert_close(author_logits, runner_logits, rtol=1e-3, atol=1e-3)
    before = adapter_fingerprint(backend.adapters.raw)
    answer = backend.answer(first, question)
    if answer["output_token_ids"] != author_tokens.tolist():
        raise AssertionError("Author and runner single-document output token IDs differ")
    if before != adapter_fingerprint(backend.adapters.raw):
        raise AssertionError("Question generation changed raw adapter")
    # A repeated bind must restore the forward before applying it again.
    backend.adapters.bind()
    repeat_logits = model.base_model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
    torch.testing.assert_close(runner_logits, repeat_logits, rtol=1e-3, atol=1e-3)
    backend.reset()
    reset_logits = model.base_model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
    torch.testing.assert_close(base_logits, reset_logits, rtol=1e-3, atol=1e-3)

    backend.location = {"group": "preflight", "step": 2}
    second_raw = backend.encode(second)
    layer_idx = list(model.hypernet.layer_indices)[0]
    name = next(iter(runner_raw))
    if name != "down_proj":
        raise ValueError("Preflight currently expects down_proj")
    chunks_tensor = torch.tensor([2], device=model.device)
    raw = {name: {key: torch.cat([runner_raw[name][key], second_raw[name][key]], dim=0)
                  for key in ("A", "B")}}
    bias = model.hypernet.get_head_bias() if model.hypernet.config.use_bias else None
    combined = model.combine_lora(raw, chunks_tensor, lora_bias=bias)[name]
    a = combined["A"][0, layer_idx].float()
    b = combined["B"][0, layer_idx].float()
    # Compare a selected matrix slice in FP32 to avoid materializing a full dense matrix.
    actual = b[:, :64].T @ a[:, :64]
    explicit = sum(pair["B"][0, layer_idx, :, :64].float().T @ pair["A"][0, layer_idx, :, :64].float()
                   for pair in (runner_raw[name], second_raw[name]))
    if bias is not None:
        explicit += bias[name]["B"][layer_idx, :, :64].float().T @ bias[name]["A"][layer_idx, :, :64].float()
    torch.testing.assert_close(actual, explicit, rtol=1e-4, atol=1e-5)
    first_update = runner_raw[name]["B"][0, layer_idx, :, :64].float().T @ runner_raw[name]["A"][0, layer_idx, :, :64].float()
    weighted_raw = {name: {"A": torch.cat([runner_raw[name]["A"] * 2, second_raw[name]["A"]], dim=0),
                           "B": raw[name]["B"]}}
    weighted_combined = model.combine_lora(weighted_raw, chunks_tensor, lora_bias=bias)[name]
    weighted_matrix = (weighted_combined["B"][0, layer_idx, :, :64].float().T @
                       weighted_combined["A"][0, layer_idx, :, :64].float())
    torch.testing.assert_close(weighted_matrix, explicit + first_update, rtol=1e-4, atol=1e-5)
    backend.reset()
    backend.adapters.append(runner_raw)
    backend.adapters.append(second_raw)
    backend.adapters.bind()
    module = get_layers(model.base_model)[layer_idx].mlp.down_proj
    generator = torch.Generator(device=model.device).manual_seed(17)
    x = torch.randn(1, 1, module.in_features, device=model.device, generator=generator, dtype=module.weight.dtype)
    observed = module(x).float()
    base = F.linear(x, module.weight, module.bias).float()
    expected_delta = (x.float() @ a.T @ b) * model.peft_config.lora_alpha
    expected = (base + expected_delta).to(module.weight.dtype).float()
    torch.testing.assert_close(observed, expected, rtol=2e-2, atol=5e-2)
    info = backend.state_info()
    if info["num_chunks"] != 2 or info["physical_rank"] != 3 * model.peft_config.r:
        raise AssertionError("Unexpected combined rank")
    backend.reset()
    return {"status": "passed", "first_doc_id": first["doc_id"], "second_doc_id": second["doc_id"],
            "selected_layer": layer_idx, "matrix_slice": [64, 64],
            "single_logit_max_abs_diff": (author_logits-runner_logits).abs().max().item(),
            "repeat_logit_max_abs_diff": (runner_logits-repeat_logits).abs().max().item(),
            "reset_logit_max_abs_diff": (base_logits-reset_logits).abs().max().item(),
            "matrix_max_abs_diff": (actual-explicit).abs().max().item(),
            "layer_output_max_abs_diff": (observed-expected).abs().max().item(),
            "raw_rtol": 1e-4, "raw_atol": 1e-5, "logits_rtol": 1e-3, "logits_atol": 1e-3,
            "matrix_rtol": 1e-4, "matrix_atol": 1e-5,
            "layer_rtol": 2e-2, "layer_atol": 5e-2}
