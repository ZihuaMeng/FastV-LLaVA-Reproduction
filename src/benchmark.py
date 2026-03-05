"""
Phase 4: Benchmark — Baseline vs FastV K=2 R=50% vs FastV K=2 R=75%

Generates a 336x336 four-quadrant test image locally (no URL downloads),
then runs three inference passes with fresh inline attention profiling for
each FastV pass (no attn_scores_layer2.pt dependency).

Metrics per pass: tokens/sec, VRAM allocated, KV-cache sequence length.
Results saved to logs/benchmark_results.json.
"""

import json
import os
import time
import urllib.request

import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)
from transformers.cache_utils import DynamicCache

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = "llava-hf/llava-1.5-7b-hf"
TEST_PROMPT    = ("USER: <image>\n"
                  "Describe the colors and shapes you see in this image.\n"
                  "ASSISTANT:")
MAX_NEW_TOKENS = 128
K              = 2      # attention-profiling layer index (0-indexed decoder layer)
N_VISUAL       = 576    # fixed for LLaVA-1.5 (336 px / 14 px patch = 24² = 576)
IMAGE_PATH     = "data/test_image.png"
RESULTS_PATH   = "logs/benchmark_results.json"
# ─────────────────────────────────────────────────────────────────────────────


# ── Image generation ──────────────────────────────────────────────────────────

def make_test_image(path: str) -> Image.Image:
    """
    Create a 336×336 image with four distinct colour quadrants and a
    white circle in the centre.  Saved to *path* and returned.
    """
    img  = Image.new("RGB", (336, 336))
    draw = ImageDraw.Draw(img)
    # Quadrants (coordinates are inclusive for PIL rectangles)
    draw.rectangle([0,   0,   167, 167], fill=(200,  50,  50))  # top-left:     red
    draw.rectangle([168, 0,   335, 167], fill=( 50,  50, 200))  # top-right:    blue
    draw.rectangle([0,   168, 167, 335], fill=( 50, 180,  50))  # bottom-left:  green
    draw.rectangle([168, 168, 335, 335], fill=(220, 200,  50))  # bottom-right: yellow
    # White circle centred at (168, 168) with radius 60
    cx, cy, r = 168, 168, 60
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    return img


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_vram_gb() -> float:
    return torch.cuda.memory_allocated() / (1024 ** 3)


def find_visual_range(input_ids: torch.Tensor, image_token_index: int):
    """
    Locate the contiguous block of image tokens in *input_ids*.
    Returns (img_start, img_end, n_visual) — img_end is exclusive.
    """
    mask      = (input_ids[0] == image_token_index)
    positions = mask.nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        raise ValueError(
            f"image_token_index={image_token_index} not found in input_ids"
        )
    img_start = positions[0].item()
    img_end   = positions[-1].item() + 1   # exclusive
    n_visual  = img_end - img_start
    return img_start, img_end, n_visual


def _make_attn_hook(captured: dict, min_seq: int):
    """
    Forward hook for a LlamaDecoderLayer.  Captures output[1] (attention
    weights, shape (batch, heads, seq, seq)) on the first (prefill) call
    only — decode steps have seq == 1 and are skipped.
    """
    def hook(module, inp, output):
        if "attn" in captured:
            return  # already have prefill; skip decode steps
        if not isinstance(output, tuple) or len(output) < 2:
            return
        attn = output[1]
        if attn is None or attn.dim() != 4 or attn.shape[2] < min_seq:
            return
        captured["attn"] = attn.detach().float().cpu()
    return hook


def compute_visual_scores(captured: dict,
                           img_start: int,
                           img_end: int) -> torch.Tensor:
    """
    FastV Eq.(3): score each visual token by the attention it receives from
    the last query position, averaged over heads.
    Returns shape (n_visual,).
    """
    attn = captured["attn"]                           # (1, heads, seq, seq)
    return attn[0, :, -1, img_start:img_end].mean(0)  # (n_visual,)


def prune_cache(cache, keep_mask: torch.Tensor):
    """
    Remove sequence positions where keep_mask is False from every layer.
    Supports DynamicCache with .layers[i].keys/.values (transformers >=4.48)
    and legacy DynamicCache with .key_cache/.value_cache lists.
    """
    if isinstance(cache, DynamicCache):
        if hasattr(cache, 'layers'):
            for layer in cache.layers:
                layer.keys   = layer.keys  [:, :, keep_mask, :]
                layer.values = layer.values[:, :, keep_mask, :]
        else:
            for i in range(len(cache.key_cache)):
                cache.key_cache[i]   = cache.key_cache[i]  [:, :, keep_mask, :]
                cache.value_cache[i] = cache.value_cache[i][:, :, keep_mask, :]
        return cache
    else:
        return tuple(
            (k[:, :, keep_mask, :], v[:, :, keep_mask, :])
            for k, v in cache
        )


def _cache_device(cache):
    """Return the device of the first key tensor regardless of cache format."""
    if isinstance(cache, DynamicCache):
        if hasattr(cache, 'layers'):
            return cache.layers[0].keys.device
        return cache.key_cache[0].device
    return cache[0][0].device


# ── Inference passes ──────────────────────────────────────────────────────────

def run_baseline(model, processor, inputs: dict, max_new_tokens: int) -> dict:
    """Standard greedy generation — no pruning."""
    n_input = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(**inputs,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=False)
    torch.cuda.synchronize()
    gen_time = time.time() - t0

    n_output = out_ids.shape[1] - n_input
    text     = processor.decode(out_ids[0][n_input:], skip_special_tokens=True)
    return dict(
        generated_text      = text,
        n_output_tokens     = n_output,
        generation_time_sec = round(gen_time, 4),
        tokens_per_sec      = round(n_output / gen_time, 2) if gen_time > 0 else 0.0,
        vram_allocated_gb   = round(get_vram_gb(), 4),
        kv_tokens           = n_input,
    )


def run_fastv(model, processor, inputs: dict, max_new_tokens: int,
              img_start: int, img_end: int, R: float) -> dict:
    """
    FastV pruned generation (K=2 attention layer, drop-ratio R).

    Step 1  Full prefill through LLaVA (vision encoder + LLM) with a
            one-shot hook on decoder layer K to capture attention weights.
    Step 2  Rank visual tokens by their average cross-head attention score
            from the last prompt position (FastV Eq. 3); keep top 1-R.
    Step 3  Prune the populated DynamicCache to remove discarded positions.
    Step 4  Autoregressive decode through the bare LLM backbone with the
            pruned (and growing) KV cache.

    Timing starts before the prefill and includes everything through the
    last decode step — this is the end-to-end wall-clock cost.
    """
    input_ids  = inputs["input_ids"]
    seq_len    = input_ids.shape[1]
    n_visual   = img_end - img_start
    keep_count = int(n_visual * (1 - R))

    torch.cuda.synchronize()
    t0 = time.time()

    # ── Full prefill: vision encoding + LLM forward with attention outputs ────
    with torch.no_grad():
        prefill_out = model(**inputs, use_cache=True, output_attentions=True)

    cache = prefill_out.past_key_values   # all seq_len positions
    # Extract attention from layer K directly from the output
    attn_layer_k = prefill_out.attentions[K].detach().float().cpu()  # (1, heads, seq, seq)
    # Seed the first decode token from the prefill's last-position logits before
    # freeing prefill_out (which holds all 32 attention matrices from
    # output_attentions=True — freeing now keeps VRAM representative of decode).
    next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del prefill_out
    torch.cuda.empty_cache()

    # ── Compute scores and build keep mask ────────────────────────────────────
    # FastV Eq.(3): score by attention from last query position, averaged over heads
    attn_scores = attn_layer_k[0, :, -1, img_start:img_end].mean(0)
    top_indices  = torch.topk(attn_scores, keep_count).indices.sort().values

    keep_mask                           = torch.zeros(seq_len, dtype=torch.bool)
    keep_mask[:img_start]               = True   # text tokens before image
    keep_mask[img_end:]                 = True   # text tokens after image
    keep_mask[img_start + top_indices]  = True   # selected visual tokens
    pruned_len = keep_mask.sum().item()

    n_text = img_start + (seq_len - img_end)
    print(f"      Pruning: {seq_len} → {pruned_len} "
          f"(visual {n_visual}→{keep_count}, text={n_text})")

    # ── Prune KV cache ────────────────────────────────────────────────────────
    cache = prune_cache(cache, keep_mask.to(_cache_device(cache)))

    # ── Manual autoregressive decode via LLM backbone ─────────────────────────
    eos_id = processor.tokenizer.eos_token_id
    gen_ids    = [next_token.item()]

    # model.model.language_model = LlamaModel (bare backbone)
    # model.lm_head = Linear
    llm          = model.model.language_model
    lm_head      = model.lm_head
    embed_tokens = llm.embed_tokens

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            embeds  = embed_tokens(next_token)
            llm_out = llm(inputs_embeds=embeds,
                          past_key_values=cache,
                          use_cache=True)
            cache       = llm_out.past_key_values
            logits      = lm_head(llm_out.last_hidden_state[:, -1, :])
            next_token  = logits.argmax(dim=-1, keepdim=True)
            tok_val     = next_token.item()
            gen_ids.append(tok_val)
            if tok_val == eos_id:
                break

    torch.cuda.synchronize()
    gen_time = time.time() - t0

    n_output = len(gen_ids)
    text     = processor.decode(gen_ids, skip_special_tokens=True)
    return dict(
        generated_text      = text,
        n_output_tokens     = n_output,
        generation_time_sec = round(gen_time, 4),
        tokens_per_sec      = round(n_output / gen_time, 2) if gen_time > 0 else 0.0,
        vram_allocated_gb   = round(get_vram_gb(), 4),
        kv_tokens           = pruned_len,
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_table(results: dict):
    col_method = 17
    col_tps    = 8
    col_vram   = 9
    col_kv     = 9

    h_sep = (f"┌{'─'*(col_method+2)}┬{'─'*(col_tps+2)}"
             f"┬{'─'*(col_vram+2)}┬{'─'*(col_kv+2)}┐")
    m_sep = h_sep.replace("┌","├").replace("┬","┼").replace("┐","┤")
    b_sep = h_sep.replace("┌","└").replace("┬","┴").replace("┐","┘")

    def row(method, tps, vram, kv, header=False):
        if header:
            return (f"│ {'Method':<{col_method}} │ {'Tok/sec':>{col_tps}} "
                    f"│ {'VRAM (GB)':>{col_vram}} │ {'KV tokens':>{col_kv}} │")
        return (f"│ {method:<{col_method}} │ {tps:>{col_tps}.1f} "
                f"│ {vram:>{col_vram}.2f} │ {kv:>{col_kv}d} │")

    print(h_sep)
    print(row(None, None, None, None, header=True))
    print(m_sep)
    for name, r in results.items():
        print(row(name, r["tokens_per_sec"], r["vram_allocated_gb"], r["kv_tokens"]))
    print(b_sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FastV Reproduction — Phase 4: Benchmark")
    print("=" * 70)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # ── Step 1: Generate test image ───────────────────────────────────────────
    print(f"\n[1/6] Generating test image → {IMAGE_PATH}")
    image = make_test_image(IMAGE_PATH)
    print(f"      Size   : {image.size}")
    print(f"      Regions: red (TL), blue (TR), green (BL), yellow (BR), white circle")

    # ── Step 2: 4-bit quantization config ────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # ── Step 3: Load model ────────────────────────────────────────────────────
    print(f"\n[2/6] Loading model: {MODEL_ID}")
    print("      attn_implementation=eager  (needed for output_attentions)")
    t0 = time.time()
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"      Load time : {time.time() - t0:.1f}s")
    print(f"      VRAM      : {get_vram_gb():.2f} GB")

    # ── Step 4: Preprocess ────────────────────────────────────────────────────
    print(f"\n[3/6] Preprocessing inputs...")
    inputs = processor(
        text=TEST_PROMPT, images=image, return_tensors="pt"
    ).to(model.device)
    input_ids = inputs["input_ids"]
    seq_len   = input_ids.shape[1]

    img_start, img_end, n_visual = find_visual_range(
        input_ids, model.config.image_token_index
    )
    print(f"      Sequence length   : {seq_len}")
    print(f"      image_token_index : {model.config.image_token_index}")
    print(f"      Visual token range: [{img_start}, {img_end})  ({n_visual} tokens)")

    # ── Step 5: Pass A — Baseline ─────────────────────────────────────────────
    print(f"\n[4/6] Pass A — Baseline  (full {seq_len} tokens, "
          f"max_new_tokens={MAX_NEW_TOKENS})")
    baseline = run_baseline(model, processor, inputs, MAX_NEW_TOKENS)
    print(f"      {baseline['n_output_tokens']} tokens  "
          f"{baseline['generation_time_sec']:.2f}s  "
          f"{baseline['tokens_per_sec']:.1f} tok/s  "
          f"VRAM={baseline['vram_allocated_gb']:.2f}GB")

    # ── Step 6: Pass B — FastV K=2 R=50% ──────────────────────────────────────
    print(f"\n[5/6] Pass B — FastV K={K} R=50%  "
          f"(keep {int(n_visual * 0.50)} visual tokens)")
    fastv_50 = run_fastv(model, processor, inputs, MAX_NEW_TOKENS,
                         img_start, img_end, R=0.50)
    print(f"      {fastv_50['n_output_tokens']} tokens  "
          f"{fastv_50['generation_time_sec']:.2f}s  "
          f"{fastv_50['tokens_per_sec']:.1f} tok/s  "
          f"VRAM={fastv_50['vram_allocated_gb']:.2f}GB")

    # ── Step 7: Pass C — FastV K=2 R=75% ──────────────────────────────────────
    print(f"\n[6/6] Pass C — FastV K={K} R=75%  "
          f"(keep {int(n_visual * 0.25)} visual tokens)")
    fastv_75 = run_fastv(model, processor, inputs, MAX_NEW_TOKENS,
                         img_start, img_end, R=0.75)
    print(f"      {fastv_75['n_output_tokens']} tokens  "
          f"{fastv_75['generation_time_sec']:.2f}s  "
          f"{fastv_75['tokens_per_sec']:.1f} tok/s  "
          f"VRAM={fastv_75['vram_allocated_gb']:.2f}GB")

    # ── Results ───────────────────────────────────────────────────────────────
    results = {
        "Baseline"        : baseline,
        "FastV K=2 R=50%" : fastv_50,
        "FastV K=2 R=75%" : fastv_75,
    }

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print_table(results)

    print("Generated text per method:")
    print()
    for method, r in results.items():
        print(f"  [{method}]")
        print(f"    {r['generated_text']}")
        print()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"  Results saved → {RESULTS_PATH}")
    print("\n✓ Phase 4 complete.")


REAL_IMAGE_PATH = "data/real_test_image.jpg"
REAL_IMAGE_URLS = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/4/43/Cute_dog.jpg",
]
REAL_TEST_PROMPT = ("USER: <image>\n"
                    "What do you see in this image?\n"
                    "ASSISTANT:")


def download_test_image() -> Image.Image:
    """Download a real test image. Falls back to synthetic if all URLs fail."""
    os.makedirs(os.path.dirname(REAL_IMAGE_PATH), exist_ok=True)
    for url in REAL_IMAGE_URLS:
        try:
            print(f"      Trying: {url}")
            urllib.request.urlretrieve(url, REAL_IMAGE_PATH)
            img = Image.open(REAL_IMAGE_PATH).convert("RGB")
            print(f"      Success! Size: {img.size}")
            return img
        except Exception as e:
            print(f"      Failed: {e}")
    print("      WARNING: All URLs failed. Using synthetic image as fallback.")
    return make_test_image(REAL_IMAGE_PATH)


def real_image_test():
    print("=" * 70)
    print("FastV Reproduction — Real Image Test")
    print("=" * 70)

    os.makedirs("logs", exist_ok=True)

    # ── Download image ───────────────────────────────────────────────────
    print("\n[1/4] Downloading real test image...")
    image = download_test_image()

    # ── Load model ───────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print(f"\n[2/4] Loading model: {MODEL_ID}")
    t0 = time.time()
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"      Load time: {time.time() - t0:.1f}s")

    # ── Preprocess ───────────────────────────────────────────────────────
    print("\n[3/4] Preprocessing...")
    inputs = processor(
        text=REAL_TEST_PROMPT, images=image, return_tensors="pt"
    ).to(model.device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    img_start, img_end, n_visual = find_visual_range(
        input_ids, model.config.image_token_index
    )
    print(f"      Sequence length: {seq_len}, visual tokens: {n_visual}")

    # ── Pass A: Baseline ─────────────────────────────────────────────────
    print(f"\n[4/4] Running inference passes...")
    print(f"  Pass A — Baseline (full {n_visual} visual tokens)")
    baseline = run_baseline(model, processor, inputs, MAX_NEW_TOKENS)
    print(f"      {baseline['tokens_per_sec']:.1f} tok/s")

    # ── Pass B: FastV K=2 R=50% ──────────────────────────────────────────
    print(f"  Pass B — FastV K={K} R=50% ({int(n_visual * 0.50)} visual tokens)")
    fastv_50 = run_fastv(model, processor, inputs, MAX_NEW_TOKENS,
                         img_start, img_end, R=0.50)
    print(f"      {fastv_50['tokens_per_sec']:.1f} tok/s")

    # ── Side-by-side output ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATED TEXT COMPARISON")
    print("=" * 70)
    print(f"\n  [Baseline]")
    print(f"    {baseline['generated_text']}")
    print(f"\n  [FastV K=2 R=50%]")
    print(f"    {fastv_50['generated_text']}")
    print()

    # ── Append to benchmark_results.json ─────────────────────────────────
    results_data = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as fh:
            results_data = json.load(fh)
    results_data["real_image_test"] = {
        "Baseline": baseline,
        "FastV K=2 R=50%": fastv_50,
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(results_data, fh, indent=2)
    print(f"  Results appended → {RESULTS_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        real_image_test()
    else:
        main()
