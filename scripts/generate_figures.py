#!/usr/bin/env python3
"""Generate publication-quality figures for FastV reproduction study."""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
LOGS = os.path.join(ROOT, "logs")
os.makedirs(ASSETS, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150


def save(fig, name):
    for ext in (".png", ".pdf"):
        fig.savefig(os.path.join(ASSETS, name + ext), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Figure 1: Attention Collapse ─────────────────────────────────────────
def fig1_attention_collapse():
    layers = [0, 1, 2, 3, 5, 10, 20, 31]
    img_attn = [0.724, 0.406, 0.169, 0.091, 0.060, 0.093, 0.078, 0.186]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, img_attn, "o-", color="steelblue", linewidth=2, markersize=7)
    ax.axhline(y=576 / 6000 * 10, color="red", linestyle="--", alpha=0.6)
    # Uniform expectation: 576/600 ≈ 0.96 → but paper uses fraction so 576/600
    uniform = 576 / 600
    ax.axhline(y=uniform, color="red", linestyle="--", alpha=0.6,
               label=f"Uniform expectation (576/600 = {uniform:.1%})")
    # Remove the first wrong line
    ax.lines[1].remove()

    ax.annotate("Shallow: 72.4%", xy=(0, 0.724), xytext=(2, 0.68),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("Deep: 9.3%", xy=(10, 0.093), xytext=(14, 0.18),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel("Transformer Layer Index", fontsize=12)
    ax.set_ylabel("Attention Allocated to Visual Tokens", fontsize=12)
    ax.set_title("Visual Token Attention Collapse Across LLaVA-1.5-7B Layers", fontsize=13)
    ax.set_xticks(layers)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    save(fig, "fig1_attention_collapse")


# ── Figure 2: Token Score Distribution ───────────────────────────────────
def fig2_token_score_distribution():
    scores = torch.load(os.path.join(LOGS, "attn_scores_layer2.pt"),
                        map_location="cpu", weights_only=True).numpy()

    p50 = np.percentile(scores, 50)
    p75 = np.percentile(scores, 75)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(p50, color="red", linestyle="--", linewidth=2, label=f"50th pctl (R=50%): {p50:.4f}")
    ax.axvline(p75, color="darkorange", linestyle="--", linewidth=2, label=f"75th pctl (R=75%): {p75:.4f}")

    ymax = ax.get_ylim()[1]
    ax.text(p50 * 0.55, ymax * 0.85, "Pruned\n(R=50%)", ha="center", fontsize=10,
            color="red", fontweight="bold")
    ax.text(p50 + (scores.max() - p50) * 0.5, ymax * 0.85, "Kept", ha="center",
            fontsize=10, color="steelblue", fontweight="bold")

    ax.set_xlabel("Attention Score", fontsize=12)
    ax.set_ylabel("Token Count", fontsize=12)
    ax.set_title("Per-Visual-Token Attention Score Distribution at Layer K=2", fontsize=13)
    ax.legend(fontsize=10)
    save(fig, "fig2_token_score_distribution")


# ── Figure 3: Benchmark Comparison ───────────────────────────────────────
def fig3_benchmark_comparison():
    labels = ["Baseline", "FastV K=2\nR=50%", "FastV K=2\nR=75%"]
    throughput = [16.8, 20.5, 19.4]
    kv_cache = [600, 312, 168]
    colors = ["gray", "steelblue", "darkorange"]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width / 2, throughput, width, color=colors, alpha=0.85, label="Throughput")
    bars2 = ax2.bar(x + width / 2, kv_cache, width, color=colors, alpha=0.45,
                    edgecolor=colors, linewidth=1.5, label="KV Cache Size")

    for bar, val in zip(bars1, throughput):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val}", ha="center", fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, kv_cache):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                 f"{val}", ha="center", fontsize=10, fontweight="bold")

    ax1.set_ylabel("Throughput (tokens/sec)", fontsize=12)
    ax2.set_ylabel("KV Cache Size (tokens)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_title("FastV Performance vs. Baseline on LLaVA-1.5-7B (RTX 5060 8GB)", fontsize=13)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    save(fig, "fig3_benchmark_comparison")


# ── Figure 4: KV Compression Ratio ──────────────────────────────────────
def fig4_kv_compression_ratio():
    configs = ["Baseline", "FastV R=50%", "FastV R=75%"]
    kept = [600, 312, 168]
    total = 600

    fig, ax = plt.subplots(figsize=(8, 3.5))

    for i, (label, k) in enumerate(zip(configs, kept)):
        saved = total - k
        ax.barh(i, k, color="steelblue", edgecolor="white")
        ax.barh(i, saved, left=k, color="#ff9999", edgecolor="white")
        pct = k / total * 100
        ax.text(k / 2, i, f"{k} tokens ({pct:.0f}%)", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        if saved > 40:
            ax.text(k + saved / 2, i, f"-{saved} saved", ha="center", va="center",
                    fontsize=9, color="#cc3333")

    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=11)
    ax.set_xlabel("Number of KV Cache Tokens", fontsize=12)
    ax.set_title("KV Cache Compression: FastV vs Baseline", fontsize=13)
    ax.legend(["Kept", "Saved"], loc="lower right", fontsize=10)
    ax.invert_yaxis()
    save(fig, "fig4_kv_compression_ratio")


# ── Figure 5: Attention Heatmap ──────────────────────────────────────────
def fig5_attention_heatmap():
    scores = torch.load(os.path.join(LOGS, "attn_scores_layer2.pt"),
                        map_location="cpu", weights_only=True).numpy()
    grid = scores[:576].reshape(24, 24)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap="hot", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Score", fontsize=11)

    ax.set_title("Spatial Distribution of Visual Token Attention at Layer K=2\n"
                 "(Lighter = Higher Attention, Kept by FastV)", fontsize=11)
    ax.set_xlabel("Patch Column", fontsize=11)
    ax.set_ylabel("Patch Row", fontsize=11)
    ax.text(0.5, -0.12, "Reshaped from 576 tokens to 24x24 grid for visualization",
            transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="gray")
    save(fig, "fig5_attention_heatmap")


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig1_attention_collapse()
    fig2_token_score_distribution()
    fig3_benchmark_comparison()
    fig4_kv_compression_ratio()
    fig5_attention_heatmap()

    # Report
    files = sorted(f for f in os.listdir(ASSETS) if f.startswith("fig"))
    print(f"\nGenerated {len(files)} figures in assets/:")
    for f in files:
        size = os.path.getsize(os.path.join(ASSETS, f))
        print(f"  {f:45s} {size / 1024:.1f} KB")
