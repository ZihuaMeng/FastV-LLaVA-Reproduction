"""
Phase 2: FastV Attention Profiling

Captures per-layer attention weights for visual tokens in LLaVA-1.5-7B using
PyTorch forward hooks. Implements equation (3) from the FastV paper:
    λ_img^j = Σ_i α_img^{i,j}   (total attention allocated to each visual token)

Key notes on LLaVA-1.5 token layout:
  - processor() produces input_ids with a single <image> placeholder token.
  - model.forward() expands that placeholder into N_VISUAL=576 patch embeddings.
  - Embedded sequence length = len(input_ids) - 1 + 576 = len(input_ids) + 575.
  - Visual tokens occupy positions [placeholder_pos, placeholder_pos + 576) in
    the embedded sequence (and thus in the attention weight matrices).
"""

import os
import time
import torch
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID     = "llava-hf/llava-1.5-7b-hf"
TEST_PROMPT  = "USER: <image>\nDescribe what you see in this image in detail.\nASSISTANT:"
TARGET_LAYERS = [0, 1, 2, 3, 5, 10, 20, 31]
N_VISUAL     = 576          # fixed patch count for LLaVA-1.5 (336px / patch_size 14)
LOG_DIR      = "logs"
# ─────────────────────────────────────────────────────────────────────────────


def get_synthetic_image() -> Image.Image:
    print("      Source: synthetic RGB image (336x336)")
    return Image.new("RGB", (336, 336), color=(128, 90, 60))


def main():
    print("=" * 65)
    print("FastV Reproduction — Phase 2: Attention Profiling")
    print("=" * 65)

    os.makedirs(LOG_DIR, exist_ok=True)

    # ── 4-bit quantization config ─────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading model: {MODEL_ID}")
    print("      attn_implementation=eager (required for output_attentions)")
    t0 = time.time()

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",   # SDPA doesn't expose attention weights
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"      Load time : {time.time() - t0:.1f}s")
    print(f"      VRAM alloc: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ── Prepare synthetic image ───────────────────────────────────────────────
    print("\n[2/5] Preparing test image...")
    image = get_synthetic_image()

    # ── Preprocess ────────────────────────────────────────────────────────────
    print("\n[3/5] Preprocessing inputs...")
    inputs = processor(
        text=TEST_PROMPT,
        images=image,
        return_tensors="pt",
    ).to(model.device)

    input_ids = inputs["input_ids"]   # shape: (1, N) — N includes 1 placeholder
    N = input_ids.shape[1]
    print(f"      input_ids length   : {N}  (includes 1 <image> placeholder)")
    print(f"      embedded seq length: {N + N_VISUAL - 1}  (placeholder → 576 patches)")

    # ── Find visual token range in EMBEDDED sequence ──────────────────────────
    image_token_index = model.config.image_token_index
    placeholder_hits = (input_ids[0] == image_token_index).nonzero(as_tuple=True)[0]

    if len(placeholder_hits) == 0:
        print(f"      WARNING: image_token_index={image_token_index} not in input_ids")
        print("      Falling back to: img_start = position 1 (after BOS)")
        img_start = 1
    else:
        img_start = placeholder_hits[0].item()

    img_end = img_start + N_VISUAL   # exclusive
    print(f"      image_token_index  : {image_token_index}")
    print(f"      Visual token range : [{img_start}, {img_end})  ({N_VISUAL} tokens)")

    # ── Register forward hooks on target transformer layers ───────────────────
    print("\n[4/5] Registering hooks and running prefill forward pass...")
    captured_attn: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        """
        LlamaDecoderLayer output with output_attentions=True:
            (hidden_states, self_attn_weights, present_key_value)
        self_attn_weights: (batch, heads, seq_len, seq_len)

        We only capture the FIRST call (prefill). Subsequent calls during
        generate() would have seq_len=1 (decode step with KV-cache).
        """
        def hook(module, inp, output):
            if layer_idx in captured_attn:
                return  # already captured prefill; skip decode steps
            if not isinstance(output, tuple) or len(output) < 2:
                return
            attn = output[1]
            if attn is None or attn.dim() != 4:
                return
            if attn.shape[2] < N_VISUAL:
                return  # decode step (seq_len=1) — skip
            captured_attn[layer_idx] = attn.detach().float().cpu()
        return hook

    hooks = []
    for idx in TARGET_LAYERS:
        layer = model.model.language_model.layers[idx]
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    # Single prefill forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    for h in hooks:
        h.remove()

    # Fallback: if hooks captured nothing, pull from outputs.attentions
    if not captured_attn:
        print("      (hooks empty — falling back to outputs.attentions)")
        if outputs.attentions is not None:
            for idx in TARGET_LAYERS:
                if idx < len(outputs.attentions):
                    a = outputs.attentions[idx]
                    if a is not None and a.dim() == 4:
                        captured_attn[idx] = a.detach().float().cpu()

    print(f"      Captured: {len(captured_attn)}/{len(TARGET_LAYERS)} target layers")

    # Sanity-check the actual embedded seq_len from first captured tensor
    if captured_attn:
        sample = next(iter(captured_attn.values()))
        actual_seq_len = sample.shape[2]
        print(f"      Actual embedded seq_len from attn: {actual_seq_len}")
        # Recompute img_end in case embedding length differs from expectation
        if img_end > actual_seq_len:
            print(f"      WARNING: img_end={img_end} > seq_len={actual_seq_len}; clamping.")
            img_end = min(img_end, actual_seq_len)

    # ── Compute per-layer visual attention — FastV eq.(3) ─────────────────────
    print("\n[5/5] Computing per-layer attention allocated to visual tokens...")
    print()
    print("=" * 65)
    print("RESULTS — Attention to Visual Tokens (FastV Figure 3 equivalent)")
    print("=" * 65)
    print(f"  {'Layer':>6}  {'Total img attn':>15}  {'Max head attn':>13}  {'Heads >1%':>10}")
    print(f"  {'-'*6}  {'-'*15}  {'-'*13}  {'-'*10}")

    attn_scores_layer2: torch.Tensor | None = None

    for layer_idx in TARGET_LAYERS:
        if layer_idx not in captured_attn:
            print(f"  {layer_idx:>6}  {'—':>15}  {'—':>13}  {'—':>10}  (not captured)")
            continue

        attn = captured_attn[layer_idx]           # (1, heads, seq_len, seq_len)
        n_heads = attn.shape[1]

        # Attention from last prompt token → each visual token
        # FastV eq.(3): avg over heads, sum over visual positions
        attn_to_img = attn[0, :, -1, img_start:img_end]  # (heads, N_VISUAL)
        avg_attn_to_img = attn_to_img.mean(dim=0)         # (N_VISUAL,)
        total_img_attn = avg_attn_to_img.sum().item()

        # Additional diagnostics
        max_head_attn = attn_to_img.sum(dim=1).max().item()  # head with most img attn
        heads_active  = (attn_to_img.sum(dim=1) > 0.01).sum().item()

        print(f"  {layer_idx:>6}  {total_img_attn:>15.6f}  {max_head_attn:>13.6f}  {heads_active:>4}/{n_heads}")

        if layer_idx == 2:
            attn_scores_layer2 = avg_attn_to_img  # (N_VISUAL,) → input to Phase 3 pruner

    print("=" * 65)
    print()
    print("  Interpretation:")
    print("  Early layers (0-3) typically show HIGH attention to visual tokens.")
    print("  Deep layers (10, 20, 31) should show NEAR-ZERO — the 'inefficiency'")
    print("  FastV exploits: once R tokens are pruned at layer K, deep layers")
    print("  never attend to low-value visual tokens.")

    # ── Save layer-2 attention scores ─────────────────────────────────────────
    if attn_scores_layer2 is not None:
        save_path = os.path.join(LOG_DIR, "attn_scores_layer2.pt")
        torch.save(attn_scores_layer2, save_path)
        print(f"\n  Saved: {save_path}")
        print(f"  Shape: {tuple(attn_scores_layer2.shape)}  "
              f"(one score per visual token; input to Phase 3 pruner)")
        print(f"  Score range: [{attn_scores_layer2.min():.6f}, "
              f"{attn_scores_layer2.max():.6f}]")
    else:
        print("\n  WARNING: Layer 2 not captured — attn_scores_layer2.pt not saved.")

    print("\n✓ Phase 2 complete.")


if __name__ == "__main__":
    main()
