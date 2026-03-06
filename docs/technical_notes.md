# Technical Notes

## Section 1 — Two-Pass vs Single-Pass FastV

### Implemented architecture (two-pass)
This repository currently implements a two-pass approximation of FastV:

1. Run a full prefill over the original sequence (text + all visual tokens).
2. Capture layer-2 attention scores over visual tokens.
3. Build a keep mask (all text tokens + top-ranked visual tokens).
4. Prune the populated KV cache.
5. Continue generation with a manual autoregressive decode loop.

This design reduces decode-step attention cost because the post-pruning KV sequence is shorter. It does not reduce prefill FLOPs, since all layers still process the full visual token set before pruning.

### Paper architecture (single-pass)
The paper's original path performs inline pruning in a single forward pass. At layer `K` (typically `K=2`), `hidden_states` is truncated inside `LlamaDecoderLayer.forward()`, and downstream layers (`K+1..L-1`) run on the shorter sequence.

This reduces both:
- Prefill FLOPs (later layers see fewer tokens during prefill)
- Decode FLOPs (KV cache remains shorter during generation)

### Why `model.generate()` is incompatible with a pre-pruned KV cache
In LLaVA/Hugging Face generation, `prepare_inputs_for_generation()` enforces internal shape consistency between current inputs, cached sequence length, and attention metadata. If `past_key_values` is externally pruned before decode, its sequence dimension no longer matches the expected progression, so generation fails at input-preparation validation.

For this reason, the current implementation bypasses `model.generate()` and drives the LLM backbone directly with a manual loop that passes the pruned cache as `past_key_values`.

### Planned single-pass implementation
The planned migration is to apply a controlled monkey-patch on `LlamaDecoderLayer.forward()` using `types.MethodType`, with shared state in a `FASTV_REGISTRY` dictionary.

`FASTV_REGISTRY` pattern:
- `original_forwards`: map layer/module ids to unpatched forward methods.
- `config`: runtime parameters (`k`, `r`, image token span, enabled flag).
- `state`: per-request pruning state (scores, selected indices, cached masks).

Execution model:
1. Before inference, patch decoder layers and store originals in `FASTV_REGISTRY["original_forwards"]`.
2. At layer `K`, compute/select kept visual indices and truncate sequence-aligned tensors in-forward.
3. Propagate truncated tensors through remaining layers in the same pass.
4. Restore original forwards after the run (or via context manager) to keep behavior reversible.

This restores the paper's intended single-pass behavior and enables true prefill + decode compute savings.

## Section 2 — Blackwell (sm_120) Compatibility Notes

### bitsandbytes issue
- Current PyPI `bitsandbytes` wheels do not ship CUDA kernels for `sm_120` (Blackwell).
- On Blackwell GPUs, 4-bit quantization paths can fail unless `bitsandbytes` is compiled from source for `sm_120`.

### Working build path
Use CMake with explicit architecture targeting:

```bash
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
cmake -DCOMPUTE_BACKEND=cuda -DSM=120 -B build
cmake --build build --config Release
pip install -e .
```

Important: do not pass `-DCUDA_VERSION`. Omitting it avoids a known CMake parsing/comparison failure and allows toolkit auto-detection.

### PyTorch dependency note
PyTorch nightly (CUDA 12.8 build) is required in this setup to match the local Blackwell toolchain/runtime combination used by this repo.

## Section 3 — Known Gaps vs Paper

| Claim | Paper | This Repo | Gap |
|---|---|---|---|
| FLOPs reduction | Single-pass inline pruning reduces compute after layer `K` during both prefill and decode. | Two-pass flow runs full prefill, then prunes cache before manual decode. | Missing prefill FLOPs reduction; only decode-side savings are measured. |
| VRAM reduction | Lower active sequence length after pruning is intended to reduce memory pressure in later computation. | Peak allocation is near baseline (or slightly higher) because full prefill cache is materialized before pruning. | No reliable peak VRAM reduction yet. |
| Decode speedup | Reports meaningful end-to-end acceleration from shorter effective sequence processing. | Decode throughput improves (e.g., ~22% at `R=50%` in local benchmark). | Partial match: decode speedup exists, but end-to-end gains are limited by unchanged prefill cost. |
| Output quality | Quality/accuracy degradation is reported as small across paper evaluations. | Only limited qualitative checks are documented; some repetition is observed under aggressive pruning/4-bit settings. | No rigorous quality parity claim can be made yet. |
| Benchmark datasets | Evaluated on standard benchmarks (for example Nocaps, A-OKVQA, MMMU). | No full benchmark-suite reproduction currently included. | Missing standardized quantitative evaluation. |
