"""
Phase 3: FastV Pruned Inference

Implements the FastV pruning pipeline from Section 4.1 of the paper:
  At layer K=2, rank visual tokens by attention score, drop the bottom R=50%,
  then continue generation with a pruned KV cache.

Two-pass approach (no model source modification):
  Pass 1: Full prefill on all 599 tokens → populates DynamicCache
  Pass 2: Prune KV cache (drop low-attention visual tokens) → generate with pruned cache

Compares baseline (full sequence) vs FastV (pruned) side-by-side.
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
from transformers.cache_utils import DynamicCache

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID      = "llava-hf/llava-1.5-7b-hf"
TEST_PROMPT   = "USER: <image>\nDescribe what you see in this image in detail.\nASSISTANT:"
MAX_NEW_TOKENS = 128
K             = 2       # pruning layer (from Phase 2)
R             = 0.50    # drop ratio: remove bottom 50% of visual tokens
ATTN_SCORES_PATH = "logs/attn_scores_layer2.pt"
# ─────────────────────────────────────────────────────────────────────────────


def get_vram_gb():
    return torch.cuda.memory_allocated() / (1024 ** 3)


def run_baseline(model, processor, inputs, max_new_tokens):
    """Standard full-sequence generation (no pruning)."""
    input_ids = inputs["input_ids"]
    n_input = input_ids.shape[1]

    torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    torch.cuda.synchronize()
    gen_time = time.time() - t0

    n_output = output_ids.shape[1] - n_input
    text = processor.decode(output_ids[0][n_input:], skip_special_tokens=True)

    return {
        "text": text,
        "n_output": n_output,
        "gen_time": gen_time,
        "tps": n_output / gen_time if gen_time > 0 else 0,
        "vram_gb": get_vram_gb(),
    }


def prune_dynamic_cache(cache, keep_mask):
    """
    Prune a DynamicCache in-place by keeping only positions where keep_mask is True.
    cache.layers[i].keys / .values have shape (batch, heads, seq_len, head_dim).
    We slice dim=2 using keep_mask (bool tensor of length seq_len).
    """
    for layer in cache.layers:
        layer.keys = layer.keys[:, :, keep_mask, :]
        layer.values = layer.values[:, :, keep_mask, :]


def run_fastv(model, processor, inputs, max_new_tokens, attn_scores, img_start, img_end, R):
    """
    FastV pruned generation:
    1. Full prefill → DynamicCache with all 599 positions
    2. Build keep_mask: all text tokens + top-(1-R) visual tokens
    3. Prune the cache
    4. Manual autoregressive decode through the language model with pruned cache

    We bypass model.generate() because LLaVA's prepare_inputs_for_generation
    doesn't support being called with a pre-populated/pruned KV cache.
    Instead we drive the LLM (model.model.language_model + model.lm_head) directly.
    """
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    n_visual = img_end - img_start
    keep_count = int(n_visual * (1 - R))

    # Rank visual tokens by attention score, keep top keep_count
    kept_indices = torch.topk(attn_scores, keep_count).indices
    kept_indices_sorted = kept_indices.sort().values

    # Build keep_mask over full sequence
    keep_mask = torch.zeros(seq_len, dtype=torch.bool)
    keep_mask[:img_start] = True          # text before image
    keep_mask[img_end:] = True            # text after image
    kept_global = img_start + kept_indices_sorted
    keep_mask[kept_global] = True         # selected visual tokens

    pruned_len = keep_mask.sum().item()
    print(f"      Pruning: {seq_len} -> {pruned_len} tokens "
          f"(kept {keep_count}/{n_visual} visual, all {img_start + seq_len - img_end} text)")

    torch.cuda.synchronize()
    t0 = time.time()

    # ── Pass 1: Full prefill through LLaVA (handles vision encoding) ──────────
    with torch.no_grad():
        prefill_out = model(**inputs, use_cache=True)

    cache = prefill_out.past_key_values  # DynamicCache, seq_len=599

    # ── Prune KV cache ────────────────────────────────────────────────────────
    device = cache.layers[0].keys.device
    keep_mask_dev = keep_mask.to(device)
    prune_dynamic_cache(cache, keep_mask_dev)

    # ── Manual autoregressive decode via the LLM backbone ─────────────────────
    # Get first generated token from prefill logits (greedy)
    next_token_id = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1,1)
    eos_token_id = processor.tokenizer.eos_token_id
    generated_ids = [next_token_id.item()]

    llm = model.model.language_model   # LlamaModel
    lm_head = model.lm_head            # Linear(4096, vocab_size)
    embed_tokens = llm.embed_tokens    # Embedding

    with torch.no_grad():
        for step in range(max_new_tokens - 1):
            # Embed the new token
            tok_embeds = embed_tokens(next_token_id)  # (1, 1, hidden_dim)

            # Forward through LlamaModel with the pruned+growing cache
            llm_out = llm(
                inputs_embeds=tok_embeds,
                past_key_values=cache,
                use_cache=True,
            )
            cache = llm_out.past_key_values
            hidden = llm_out.last_hidden_state  # (1, 1, hidden_dim)

            # Project to vocabulary
            logits = lm_head(hidden[:, -1, :])  # (1, vocab_size)
            next_token_id = logits.argmax(dim=-1, keepdim=True)  # (1, 1)

            token_val = next_token_id.item()
            generated_ids.append(token_val)

            if token_val == eos_token_id:
                break

    torch.cuda.synchronize()
    gen_time = time.time() - t0

    n_output = len(generated_ids)
    text = processor.decode(generated_ids, skip_special_tokens=True)

    return {
        "text": text,
        "n_output": n_output,
        "gen_time": gen_time,
        "tps": n_output / gen_time if gen_time > 0 else 0,
        "vram_gb": get_vram_gb(),
        "pruned_len": pruned_len,
    }


def main():
    print("=" * 70)
    print("FastV Reproduction — Phase 3: FastV Pruned Inference")
    print("=" * 70)

    # ── Load attention scores from Phase 2 ────────────────────────────────────
    print(f"\n[1/6] Loading attention scores from {ATTN_SCORES_PATH}...")
    if not os.path.exists(ATTN_SCORES_PATH):
        print("      ERROR: attn_scores_layer2.pt not found. Run fastv_profiler.py first.")
        return
    attn_scores = torch.load(ATTN_SCORES_PATH, weights_only=True)
    print(f"      Shape: {tuple(attn_scores.shape)}, range: [{attn_scores.min():.6f}, {attn_scores.max():.6f}]")

    # ── 4-bit quantization config ─────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[2/6] Loading model: {MODEL_ID}")
    t0 = time.time()
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"      Load time: {time.time() - t0:.1f}s")
    print(f"      VRAM: {get_vram_gb():.2f} GB")

    # ── Prepare image + inputs ────────────────────────────────────────────────
    print("\n[3/6] Preparing inputs (synthetic image)...")
    image = Image.new("RGB", (336, 336), color=(128, 90, 60))
    inputs = processor(text=TEST_PROMPT, images=image, return_tensors="pt").to(model.device)

    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    print(f"      Sequence length: {seq_len}")

    # ── Find visual token range ───────────────────────────────────────────────
    image_token_index = model.config.image_token_index
    img_mask = (input_ids[0] == image_token_index)
    img_positions = img_mask.nonzero(as_tuple=True)[0]
    img_start = img_positions[0].item()
    img_end = img_positions[-1].item() + 1
    n_visual = img_end - img_start
    print(f"      Visual tokens: [{img_start}, {img_end}) = {n_visual} tokens")

    assert attn_scores.shape[0] == n_visual, (
        f"attn_scores shape {attn_scores.shape} != n_visual {n_visual}")

    # ── Run baseline ──────────────────────────────────────────────────────────
    print(f"\n[4/6] Running BASELINE (full sequence, max_new_tokens={MAX_NEW_TOKENS})...")
    baseline = run_baseline(model, processor, inputs, MAX_NEW_TOKENS)
    print(f"      Done: {baseline['n_output']} tokens in {baseline['gen_time']:.2f}s")

    # ── Run FastV ─────────────────────────────────────────────────────────────
    print(f"\n[5/6] Running FASTV K={K} R={R:.0%} (max_new_tokens={MAX_NEW_TOKENS})...")
    fastv = run_fastv(model, processor, inputs, MAX_NEW_TOKENS,
                      attn_scores, img_start, img_end, R)
    print(f"      Done: {fastv['n_output']} tokens in {fastv['gen_time']:.2f}s")

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n[6/6] Results comparison")
    print()
    print("=" * 70)
    print("RESULTS — Baseline vs FastV")
    print("=" * 70)
    print()
    print(f"  {'':25s}  {'BASELINE':>14s}  {'FastV K=2 R=50%':>16s}")
    print(f"  {'-'*25}  {'-'*14}  {'-'*16}")
    print(f"  {'KV cache seq_len':25s}  {seq_len:>14d}  {fastv['pruned_len']:>16d}")
    print(f"  {'Visual tokens kept':25s}  {n_visual:>14d}  {int(n_visual * (1-R)):>16d}")
    print(f"  {'Output tokens':25s}  {baseline['n_output']:>14d}  {fastv['n_output']:>16d}")
    print(f"  {'Generation time (s)':25s}  {baseline['gen_time']:>14.2f}  {fastv['gen_time']:>16.2f}")
    print(f"  {'Speed (tokens/sec)':25s}  {baseline['tps']:>14.1f}  {fastv['tps']:>16.1f}")
    print(f"  {'VRAM allocated (GB)':25s}  {baseline['vram_gb']:>14.2f}  {fastv['vram_gb']:>16.2f}")
    print()
    print(f"  BASELINE text:")
    print(f"    {baseline['text']}")
    print()
    print(f"  FASTV text:")
    print(f"    {fastv['text']}")
    print()

    # Speedup summary
    if baseline['gen_time'] > 0:
        speedup = fastv['tps'] / baseline['tps'] if baseline['tps'] > 0 else 0
        cache_reduction = 1 - (fastv['pruned_len'] / seq_len)
        print(f"  KV cache reduction : {cache_reduction:.1%}")
        print(f"  Throughput ratio   : {speedup:.2f}x")

    print("=" * 70)
    print("\n✓ Phase 3 complete.")


if __name__ == "__main__":
    main()
