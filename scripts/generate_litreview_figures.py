#!/usr/bin/env python3
"""
scripts/generate_litreview_figures.py
Generates publication-quality figures for the Literature Review:
1. PRISMA 2020 Systematic Search Flow Diagram
2. Architectural Taxonomy: Cascaded vs Audio LLM vs GuimLab Continuous Substrate
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Setup publication styling
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.titlesize': 13,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

os.makedirs("figures", exist_ok=True)

def generate_prisma_diagram():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Color definitions (accessible academic palette)
    c_id = '#e0f3f8'
    c_screen = '#fee090'
    c_inc = '#e0f3db'
    c_exc = '#fc8d59'
    border_col = '#2c3e50'

    # 1. Identification
    box_id = patches.FancyBboxPatch((0.05, 0.78), 0.90, 0.16, boxstyle="round,pad=0.03",
                                    facecolor=c_id, edgecolor=border_col, linewidth=1.5)
    ax.add_patch(box_id)
    ax.text(0.50, 0.89, "IDENTIFICATION", ha='center', va='center', fontweight='bold', fontsize=12)
    ax.text(0.50, 0.83, "Records identified from databases (n = 148)\n[arXiv (n=54), OpenAlex (n=42), CrossRef (n=31), Semantic Scholar (n=21)]",
            ha='center', va='center', fontsize=9.5)

    # Arrow 1
    ax.annotate('', xy=(0.50, 0.69), xytext=(0.50, 0.78),
                arrowprops=dict(arrowstyle="->", lw=2, color=border_col))

    # 2. Screening
    box_dup = patches.FancyBboxPatch((0.15, 0.55), 0.70, 0.14, boxstyle="round,pad=0.03",
                                     facecolor=c_screen, edgecolor=border_col, linewidth=1.5)
    ax.add_patch(box_dup)
    ax.text(0.50, 0.64, "SCREENING & DEDUPLICATION", ha='center', va='center', fontweight='bold', fontsize=12)
    ax.text(0.50, 0.58, "Records after duplicates removed (n = 94)\nDuplicates removed by DOI/Title match (n = 54)",
            ha='center', va='center', fontsize=9.5)

    # Arrow 2 & Exclusion 1
    ax.annotate('', xy=(0.50, 0.44), xytext=(0.50, 0.55),
                arrowprops=dict(arrowstyle="->", lw=2, color=border_col))
    
    box_exc1 = patches.FancyBboxPatch((0.62, 0.42), 0.35, 0.10, boxstyle="round,pad=0.02",
                                      facecolor=c_exc, edgecolor=border_col, linewidth=1.2, alpha=0.9)
    ax.add_patch(box_exc1)
    ax.text(0.795, 0.47, "Records Excluded (n = 73)\n• Off-topic / Non-speech: 38\n• High-latency discrete: 22\n• Lack theoretical formalism: 13",
            ha='center', va='center', fontsize=8.5)
    ax.annotate('', xy=(0.62, 0.48), xytext=(0.50, 0.48),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=border_col, linestyle='--'))

    # 3. Eligibility
    box_elig = patches.FancyBboxPatch((0.10, 0.30), 0.48, 0.14, boxstyle="round,pad=0.03",
                                      facecolor=c_screen, edgecolor=border_col, linewidth=1.5)
    ax.add_patch(box_elig)
    ax.text(0.34, 0.39, "FULL-TEXT ELIGIBILITY", ha='center', va='center', fontweight='bold', fontsize=12)
    ax.text(0.34, 0.33, "Full-text articles assessed\nfor eligibility (n = 21)",
            ha='center', va='center', fontsize=9.5)

    # Arrow 3
    ax.annotate('', xy=(0.34, 0.19), xytext=(0.34, 0.30),
                arrowprops=dict(arrowstyle="->", lw=2, color=border_col))

    # 4. Inclusion
    box_inc = patches.FancyBboxPatch((0.05, 0.04), 0.90, 0.15, boxstyle="round,pad=0.03",
                                     facecolor=c_inc, edgecolor=border_col, linewidth=1.8)
    ax.add_patch(box_inc)
    ax.text(0.50, 0.14, "SYNTHESIZED IN SYSTEMATIC REVIEW (n = 21)", ha='center', va='center', fontweight='bold', fontsize=12, color='#00441b')
    ax.text(0.50, 0.08, "• Axis 1: Speech-to-Speech Dialogue (n=7)   • Axis 2: Continual Backpropagation & Plasticity (n=4)\n• Axis 3: Continuous Traces & Meta-Descent (n=5)   • Axis 4: GPU Substrates & Associative Memory (n=5)",
            ha='center', va='center', fontsize=9, fontweight='medium')

    plt.title("PRISMA 2020 Flow Diagram: Systematic Literature Search for GuimLab", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig("figures/litreview_prisma_flow.png")
    fig.savefig("figures/litreview_prisma_flow.svg")
    plt.close(fig)
    print("Generated figures/litreview_prisma_flow.[png,svg]")

def generate_architectural_taxonomy():
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)

    # Paradigm 1: Cascaded Pipeline
    ax1 = axes[0]
    ax1.set_xlim(0, 1500)
    ax1.set_ylim(-0.5, 1.5)
    ax1.axis('off')
    ax1.text(0, 1.1, "1. Cascaded Discrete Pipeline (ASR + LLM + TTS)", fontsize=11, fontweight='bold', color='#b2182b')
    
    # Blocks
    ax1.add_patch(patches.Rectangle((0, 0.1), 300, 0.7, facecolor='#fddbc7', edgecolor='#b2182b', lw=1.5))
    ax1.text(150, 0.45, "Whisper ASR\n(300 ms)", ha='center', va='center', fontsize=9)

    ax1.add_patch(patches.Rectangle((330, 0.1), 600, 0.7, facecolor='#fddbc7', edgecolor='#b2182b', lw=1.5))
    ax1.text(630, 0.45, "Auto-Regressive LLM\n(600 ms, Token Serialization)", ha='center', va='center', fontsize=9)

    ax1.add_patch(patches.Rectangle((960, 0.1), 350, 0.7, facecolor='#fddbc7', edgecolor='#b2182b', lw=1.5))
    ax1.text(1135, 0.45, "Neural Vocoder / TTS\n(350 ms)", ha='center', va='center', fontsize=9)
    ax1.text(1330, 0.45, "Total: ~1250 ms", ha='left', va='center', fontweight='bold', color='#b2182b')

    # Paradigm 2: Audio Foundation LLM
    ax2 = axes[1]
    ax2.set_ylim(-0.5, 1.5)
    ax2.axis('off')
    ax2.text(0, 1.1, "2. End-to-End Duplex Audio LLMs (Moshi, Mini-Omni, Llama-Omni)", fontsize=11, fontweight='bold', color='#2166ac')
    
    ax2.add_patch(patches.Rectangle((0, 0.1), 80, 0.7, facecolor='#d1e5f0', edgecolor='#2166ac', lw=1.5))
    ax2.text(40, 0.45, "Codec\n(80 ms)", ha='center', va='center', fontsize=8.5)

    ax2.add_patch(patches.Rectangle((90, 0.1), 160, 0.7, facecolor='#d1e5f0', edgecolor='#2166ac', lw=1.5))
    ax2.text(170, 0.45, "Duplex 7B LLM (160 ms)", ha='center', va='center', fontsize=8.5)
    ax2.text(280, 0.45, "Total: ~240 ms (Python/PyTorch Bound)", ha='left', va='center', fontweight='bold', color='#2166ac')

    # Paradigm 3: GuimLab Bare-Metal Substrate
    ax3 = axes[2]
    ax3.set_ylim(-0.5, 1.5)
    ax3.axis('off')
    ax3.text(0, 1.1, "3. GuimLab Continuous Neuromorphic Substrate (Pure CUDA/C++20)", fontsize=11, fontweight='bold', color='#1b9e77')

    ax3.add_patch(patches.Rectangle((0, 0.1), 0.31, 0.7, facecolor='#b4e2d3', edgecolor='#1b9e77', lw=2))
    ax3.text(50, 0.45, "Host-GPU Round-Trip (p50: 0.308 ms | Kernel Compute: 0.0038 ms)", ha='left', va='center', fontweight='bold', fontsize=9.5, color='#1b9e77')
    ax3.text(800, 0.45, "Zero Allocations | Plasticity Preservation (CBP) | Symplectic Traces", ha='left', va='center', fontsize=9, style='italic')

    # Bottom axis marker
    ax3.axhline(-0.2, color='black', lw=1.5)
    for pos, label in [(0, "0 ms"), (300, "300 ms"), (600, "600 ms"), (900, "900 ms"), (1200, "1200 ms"), (1500, "1500 ms")]:
        ax3.plot([pos, pos], [-0.2, -0.1], color='black', lw=1.5)
        ax3.text(pos, -0.4, label, ha='center', va='top', fontsize=8.5)

    plt.suptitle("Architectural Taxonomy & Latency Spectrum for Spoken Conversational Intelligence", fontsize=12.5, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig("figures/litreview_architectural_taxonomy.png")
    fig.savefig("figures/litreview_architectural_taxonomy.svg")
    plt.close(fig)
    print("Generated figures/litreview_architectural_taxonomy.[png,svg]")

if __name__ == "__main__":
    generate_prisma_diagram()
    generate_architectural_taxonomy()
