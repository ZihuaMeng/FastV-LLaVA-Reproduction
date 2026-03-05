# FastV Reproduction on LLaVA-1.5-7B

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-nightly%20cu128-ee4c2c?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

A pure inference-time reproduction of the **FastV** paper:

> **"An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Acceleration for VLLM Inference"**  
> Chen et al., arXiv:2403.06764, 2024

---

## Overview

FastV is a training-free acceleration method for vision-language models (VLLMs).
It observes that attention allocated to visual tokens collapses sharply after the
second transformer layer: deep layers spend most of their compute attending to text
while nearly ignoring the image.
FastV exploits this by profiling per-token attention scores at layer K (default K = 2),
ranking the visual tokens, discarding the bottom R% of them, and continuing
generation with a pruned KV cache — achieving a significant reduction in both
FLOPs and memory bandwidth without re-training the model.

**Goal of this repository:** reproduce the core FastV results from scratch on
`llava-hf/llava-1.5-7b-hf` with a 4-bit INT4 quantized model running under
8 GB VRAM.
No fine-tuning, dataset preparation, or training infrastructure is involved —
every script is a self-contained inference pass.

---

## Hardware & Environment

| Component | Detail |
|-----------|--------|
| OS | Windows 11 + WSL2 Ubuntu 24.04 |
| GPU | NVIDIA RTX 5060 8 GB VRAM (Blackwell sm_120) |
| Python | 3.11 |
| PyTorch | nightly (CUDA 12.8) |
| transformers | 4.46.3 |
| bitsandbytes | 0.50.0.dev0 (compiled from source for sm_120) |

---

## Installation

### 1. Create the conda environment

```bash
conda create -n fastv python=3.11 -y
conda activate fastv
```

### 2. Install PyTorch nightly (CUDA 12.8)

```bash
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 3. Build bitsandbytes from source (required for Blackwell sm_120)

The PyPI wheel does not include sm_120 kernels. Compile from source and pass
`-DSM=120` to target the Blackwell architecture.

> **Important:** omit `-DCUDA_VERSION` entirely — passing it as a string value
> triggers a CMake numeric-comparison parsing bug. CMake reads the CUDA version
> automatically from the toolkit.

```bash
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes

cmake -DCOMPUTE_BACKEND=cuda -DSM=120 -B build
cmake --build build --config Release
pip install -e .
cd ..
```

Verify the build:

```bash
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

### 4. Install remaining dependencies

```bash
pip install transformers==4.46.3 accelerate pillow
```

---

## Project Structure

```
fastv_reproduce/
├── src/
│   ├── baseline_inference.py   # Phase 1 — baseline LLaVA generation, VRAM/speed metrics
│   ├── fastv_profiler.py       # Phase 2 — per-layer attention profiling (FastV Figure 3)
│   ├── fastv_inference.py      # Phase 3 — FastV pruning + KV cache realignment
│   └── benchmark.py            # Phase 4 — three-way comparison on a real test image
├── logs/
│   ├── attn_scores_layer2.pt   # saved pruning signal (576-dim attention scores, layer 2)
│   └── benchmark_results.json  # JSON output from benchmark.py
├── data/
│   └── test_image.png          # locally generated 336×336 four-quadrant test image
├── configs/
└── notebooks/
```

Each script is runnable standalone:

```bash
conda activate fastv
python src/baseline_inference.py   # ~17 tok/s on RTX 5060 INT4
python src/fastv_profiler.py       # captures layer-2 attention scores
python src/fastv_inference.py      # pruned vs baseline side-by-side
python src/benchmark.py            # full three-way benchmark table + JSON
```

---

## Reproduction Results

### Attention Profiling (reproduces paper Figure 3)

`fastv_profiler.py` computes the fraction of total attention allocated to
visual tokens at each transformer layer (FastV Eq. 3).

| Layer | Total img attn | Paper finding |
|------:|---------------:|---------------|
| 0 | 72.4% | Shallow layer: attention broadly distributed over image |
| 2 | 16.9% | Beginning to collapse — FastV prunes here |
| 3–20 | 6–9% | Deep layers: extremely inefficient use of visual tokens |
| 31 | 18.6% | Slight rebound at final layer |

This matches the key finding in the paper: deep layers largely ignore the image,
making it safe to drop low-attention visual tokens after layer 2.

---

### Benchmark Results

Three-way comparison produced by `benchmark.py` on the locally generated
336×336 four-quadrant image (red / blue / green / yellow quadrants,
white circle in centre).

| Method | Tok/sec | VRAM | KV Tokens |
|--------|--------:|-----:|----------:|
| Baseline | 16.8 | 3.84 GB | 600 |
| FastV K=2 R=50% | 20.5 | 4.06 GB | 312 |
| FastV K=2 R=75% | 19.4 | 3.97 GB | 168 |

Key observations:
- **+22% throughput** at R=50%: the pruned decode loop operates over a shorter
  KV cache, reducing attention computation per decode step.
- Visual token count halves (576 → 288) or quarters (576 → 144) as expected.
- VRAM does not decrease below baseline because this two-pass approach stores the
  full KV cache during prefill before pruning (see [Implementation Notes](#implementation-notes)).

---

### Semantic Preservation (real-image test)

Tested on a Volkswagen Beetle photograph with
`"Describe what you see in this image in detail."`:

| Method | Generated text (excerpt) |
|--------|--------------------------|
| Baseline | "A vintage blue Volkswagen Beetle parked on a street…" |
| FastV K=2 R=50% | "The car on the front of a blue car…" (subject preserved) |

The pruned model retains the core semantic content (blue car, street scene)
while dropping fine-grained detail — consistent with the paper's claim that
FastV preserves answer accuracy for most VQA tasks.

---

## Implementation Notes

Two engineering findings arose during implementation that are **not described
in the paper**:

### a) `model.generate()` is incompatible with a pre-pruned KV cache

LLaVA's `prepare_inputs_for_generation()` performs internal shape validation
on `past_key_values` that rejects any cache whose sequence dimension has been
modified.
Passing a pruned cache directly causes a shape mismatch error at the first
decode step.

**Solution:** bypass `model.generate()` entirely and drive the LLM backbone
(`model.language_model.model`) in a manual autoregressive loop, passing the
pruned cache as `past_key_values` and forwarding one token embedding at a time.

### b) Two-pass approach vs. the paper's single-pass inline pruning

The paper's reference implementation truncates `hidden_states` at layer K
inside the forward pass, so layers K+1 … 31 never process the dropped tokens.
This is the true source of FLOPs reduction.

This repository uses a **two-pass approach**:

1. Full prefill through all 32 layers → populates the KV cache for all 600 positions.
2. Prune the populated KV cache (drop low-attention visual positions).
3. Autoregressive decode with the pruned (and growing) cache.

The prefill FLOPs are therefore identical to baseline.
Only the per-decode-step attention cost is reduced by the shorter KV sequence.
This is a valid approximation for measuring decode throughput, but does not
reproduce the paper's full computational savings.

---

## Known Limitations

- **Repetition in FastV output:** the manual decode loop does not apply
  `repetition_penalty`; combined with 4-bit quantization, aggressive pruning
  (R=75%) can cause the model to fall into repetitive loops.
- **VRAM does not decrease vs baseline:** two-pass profiling stores the full
  KV cache before pruning; peak allocation equals the unpruned baseline.
- **No standard benchmarks evaluated:** VQA accuracy across Nocaps, A-OKVQA,
  or MMMU datasets has not been measured in this reproduction.

---

## Citation

```bibtex
@article{chen2024fastv,
  title   = {An Image is Worth 1/2 Tokens After Layer 2:
             Plug-and-Play Acceleration for VLLM Inference},
  author  = {Chen, Liang and Zhao, Haozhe and Liu, Tianyu and others},
  journal = {arXiv:2403.06764},
  year    = {2024}
}
```
