# FastV-LLaVA Reproduction

Pure code reproduction of **"An Image is Worth 1/2 Tokens After Layer 2"** (FastV)
on `llava-hf/llava-1.5-7b-hf` with INT4 quantization under 8GB VRAM.

## Environment
- WSL2 Ubuntu 24.04 | RTX 5060 8GB VRAM
- PyTorch + Hugging Face Transformers + bitsandbytes INT4

## Phases
- [x] Phase 1: Environment & Baseline Inference
- [ ] Phase 2: FastV Hook & Attention Profiling
- [ ] Phase 3: Token Pruning + KV Cache Realignment
- [ ] Phase 4: Benchmarking
