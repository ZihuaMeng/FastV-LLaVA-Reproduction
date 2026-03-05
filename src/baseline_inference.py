"""
Phase 1: Baseline Inference
Load LLaVA-1.5-7B in 4-bit INT4 and run a single image-text inference.
Measures: VRAM usage, load time, generation speed (tokens/sec).
"""

import time
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
TEST_PROMPT = "USER: <image>\nDescribe what you see in this image in detail.\nASSISTANT:"
MAX_NEW_TOKENS = 128
# ─────────────────────────────────────────────────────────────────────────────


def load_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print("      Source      : downloaded from URL")
        return image
    except Exception as e:
        print(f"      URL download failed ({e}); using synthetic image instead")
        image = Image.new("RGB", (336, 336), color=(128, 90, 60))
        print("      Source      : synthetic RGB image (336x336)")
        return image


def get_vram_usage_gb() -> float:
    return torch.cuda.memory_allocated() / (1024 ** 3)


def get_vram_reserved_gb() -> float:
    return torch.cuda.memory_reserved() / (1024 ** 3)


def main():
    print("=" * 60)
    print("FastV Reproduction — Phase 1: Baseline Inference")
    print("=" * 60)

    # ── 4-bit quantization config ─────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,       # nested quantization saves ~0.4GB
        bnb_4bit_quant_type="nf4",            # NormalFloat4: best quality/perf ratio
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading model: {MODEL_ID}")
    print("      (First run will download ~4GB — subsequent runs use HF cache)")
    t0 = time.time()

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",          # auto-routes layers across GPU/CPU
        low_cpu_mem_usage=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    load_time = time.time() - t0

    vram_after_load = get_vram_usage_gb()
    print(f"      Load time : {load_time:.1f}s")
    print(f"      VRAM used : {vram_after_load:.2f} GB (allocated)")
    print(f"      VRAM rsrv : {get_vram_reserved_gb():.2f} GB (reserved)")

    # ── Load test image ───────────────────────────────────────────────────────
    print("\n[2/4] Loading test image from URL...")
    image = load_image_from_url(TEST_IMAGE_URL)
    print(f"      Image size: {image.size}")

    # ── Preprocess ────────────────────────────────────────────────────────────
    print("\n[3/4] Preprocessing inputs...")
    inputs = processor(
        text=TEST_PROMPT,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    input_ids = inputs["input_ids"]
    print(f"      Input token count : {input_ids.shape[1]}")

    # ── Generate ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] Generating (max_new_tokens={MAX_NEW_TOKENS})...")
    torch.cuda.synchronize()
    t_gen_start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # greedy decoding — deterministic baseline
        )

    torch.cuda.synchronize()
    gen_time = time.time() - t_gen_start

    # ── Decode & report ───────────────────────────────────────────────────────
    n_input_tokens = input_ids.shape[1]
    n_output_tokens = output_ids.shape[1] - n_input_tokens
    tokens_per_sec = n_output_tokens / gen_time

    generated_text = processor.decode(
        output_ids[0][n_input_tokens:],
        skip_special_tokens=True
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Generated text  : {generated_text}")
    print(f"  Output tokens   : {n_output_tokens}")
    print(f"  Generation time : {gen_time:.2f}s")
    print(f"  Speed           : {tokens_per_sec:.1f} tokens/sec  ← baseline target")
    print(f"  VRAM (alloc)    : {get_vram_usage_gb():.2f} GB")
    print(f"  VRAM (reserved) : {get_vram_reserved_gb():.2f} GB")
    print("=" * 60)
    print("\n✓ Phase 1 complete. Paste this output to proceed to Phase 2.")


if __name__ == "__main__":
    main()
