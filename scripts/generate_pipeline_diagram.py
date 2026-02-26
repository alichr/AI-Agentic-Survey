#!/usr/bin/env python3
"""Generate a presentation-ready PDF diagram of the 8-stage multi-view clustering pipeline."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 20, 26
BOX_W, BOX_H = 3.6, 1.1
SMALL_BOX_W, SMALL_BOX_H = 2.8, 1.0
VIEW_BOX_W, VIEW_BOX_H = 1.9, 0.68

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
C_BLUE1 = "#3B82F6"
C_BLUE2 = "#2563EB"
C_BLUE3 = "#1D4ED8"
C_PURPLE1 = "#8B5CF6"
C_PURPLE2 = "#7C3AED"
C_ORANGE1 = "#F59E0B"
C_ORANGE2 = "#D97706"
C_GREEN1 = "#10B981"
C_GREEN2 = "#059669"
C_GREEN3 = "#047857"
C_PINK = "#EC4899"
C_SIDE = "#94A3B8"
C_VIEW = "#6366F1"
TEXT_WHITE = "#FFFFFF"
TEXT_DARK = "#1E293B"
BG_COLOR = "#F8FAFC"


def _box(ax, cx, cy, w, h, color, top, bot=None,
         text_color=TEXT_WHITE, fs_top=11, fs_bot=9):
    """Rounded rectangle with up to two text lines."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="white", linewidth=1.4, zorder=3,
    )
    ax.add_patch(box)
    y_top = cy + 0.15 if bot else cy
    ax.text(cx, y_top, top, ha="center", va="center",
            fontsize=fs_top, fontweight="bold", color=text_color, zorder=4)
    if bot:
        ax.text(cx, cy - 0.17, bot, ha="center", va="center",
                fontsize=fs_bot, color=text_color, style="italic",
                zorder=4, alpha=0.92)


def _arrow(ax, x1, y1, x2, y2, label=None, color="#555555"):
    """Arrow between two points with optional label."""
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", color=color, linewidth=1.6,
        mutation_scale=16, connectionstyle="arc3,rad=0", zorder=2,
    )
    ax.add_patch(arr)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.15, my + 0.08, label, ha="left", va="center",
                fontsize=8.5, color="#64748B", style="italic", zorder=5)


def _side_box(ax, cx, cy, text, anchor_x, anchor_y, color=C_SIDE):
    """Dashed annotation box pointing to the pipeline."""
    box = FancyBboxPatch(
        (cx - 1.5, cy - 0.35), 3.0, 0.7,
        boxstyle="round,pad=0.12",
        facecolor="#F1F5F9", edgecolor=color, linewidth=1.3,
        linestyle="--", zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=9, color="#475569", zorder=4)
    ax.annotate("", xy=(anchor_x, anchor_y), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                linewidth=1.0, linestyle="--"), zorder=2)


def generate():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    MID = FIG_W / 2

    # ── Title ─────────────────────────────────────────────────────────
    ax.text(MID, FIG_H - 0.8,
            "Multi-View Clustering Pipeline for Agentic AI Survey",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color=TEXT_DARK, zorder=5)
    ax.text(MID, FIG_H - 1.4,
            "8-Stage Paper Selection System  |  5 Section Views  |  Raw Signal Output",
            ha="center", va="center", fontsize=13, color="#64748B", zorder=5)

    # ── Y positions ───────────────────────────────────────────────────
    y1 = 22.8
    y2 = 21.0
    y3 = 19.2
    y4a = 17.2
    y4b = 15.4
    y_filt = 13.9
    y5 = 12.3
    y_views = 10.7
    y6 = 9.1
    y_conv = 7.9
    y7 = 6.4
    y8 = 4.2

    # ==================================================================
    # Stage 1 — Discovery
    # ==================================================================
    _box(ax, MID, y1, BOX_W, BOX_H, C_BLUE1,
         "1  Discovery", "Scan papers/<Venue>/*.pdf")
    _arrow(ax, MID, y1 - BOX_H / 2, MID, y2 + BOX_H / 2,
           label="Paper objects")

    # ==================================================================
    # Stage 2 — Full-Text Extraction
    # ==================================================================
    _box(ax, MID, y2, BOX_W, BOX_H, C_BLUE2,
         "2  Full-Text Extraction", "PyMuPDF (all pages)")
    _arrow(ax, MID, y2 - BOX_H / 2, MID, y3 + BOX_H / 2,
           label="raw text")

    # ==================================================================
    # Stage 3 — Metadata Extraction
    # ==================================================================
    _box(ax, MID, y3, BOX_W, BOX_H, C_BLUE3,
         "3  Metadata Extraction", "Qwen3-30B via vLLM")
    _arrow(ax, MID, y3 - BOX_H / 2, MID, y4a + BOX_H / 2,
           label="title, abstract, authors, year")

    # ==================================================================
    # Stage 4a — Section Splitting
    # ==================================================================
    _box(ax, MID, y4a, BOX_W, BOX_H, C_PURPLE1,
         "4a  Section Splitting", "Regex + LLM fallback")
    _arrow(ax, MID, y4a - BOX_H / 2, MID, y4b + BOX_H / 2,
           label="5 raw sections")

    # ==================================================================
    # Stage 4b — Section Summarization
    # ==================================================================
    _box(ax, MID, y4b, BOX_W, BOX_H, C_PURPLE2,
         "4b  Summarization", "~400-word summaries (vLLM)")
    _arrow(ax, MID, y4b - BOX_H / 2, MID, y_filt + 0.25)

    # ── Filter incomplete papers ──────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (MID - 1.4, y_filt - 0.22), 2.8, 0.44,
        boxstyle="round,pad=0.10", facecolor="#FEF3C7",
        edgecolor="#F59E0B", linewidth=1.2, zorder=3,
    ))
    ax.text(MID, y_filt, "Filter incomplete papers",
            ha="center", va="center", fontsize=9,
            color="#92400E", fontweight="bold", zorder=4)
    _arrow(ax, MID, y_filt - 0.25, MID, y5 + BOX_H / 2)

    # ==================================================================
    # Stage 5 — Multi-View Embedding
    # ==================================================================
    _box(ax, MID, y5, BOX_W, BOX_H, C_ORANGE1,
         "5  Multi-View Embedding", "Qwen3-Embedding-4B",
         text_color=TEXT_DARK)

    # ── Fan-out to 5 views ────────────────────────────────────────────
    view_names = ["Title+Abs\n+Conclusion", "Intro", "Related\nWork",
                  "Method", "Experiments"]
    view_xs = [MID - 6, MID - 3, MID, MID + 3, MID + 6]

    for i, (vx, vn) in enumerate(zip(view_xs, view_names)):
        _arrow(ax, MID, y5 - BOX_H / 2, vx, y_views + VIEW_BOX_H / 2 + 0.03,
               color=C_VIEW)
        box = FancyBboxPatch(
            (vx - VIEW_BOX_W / 2, y_views - VIEW_BOX_H / 2),
            VIEW_BOX_W, VIEW_BOX_H,
            boxstyle="round,pad=0.08",
            facecolor=C_VIEW, edgecolor="white", linewidth=1.0, zorder=3,
        )
        ax.add_patch(box)
        ax.text(vx, y_views, f"V{i}: {vn}", ha="center", va="center",
                fontsize=7, fontweight="bold", color=TEXT_WHITE,
                zorder=4, linespacing=0.9)

    # ── "5 Section Views" label ───────────────────────────────────────
    ax.text(MID, y_views + VIEW_BOX_H / 2 + 0.38,
            "5 Independent Section Views",
            ha="center", va="center", fontsize=11,
            color=C_VIEW, fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=C_VIEW, linewidth=1.2))

    # ==================================================================
    # Stage 6 — K-Means Clustering (per view)
    # ==================================================================
    for vx in view_xs:
        _arrow(ax, vx, y_views - VIEW_BOX_H / 2, vx,
               y6 + VIEW_BOX_H / 2 + 0.03, color=C_ORANGE2)
        box = FancyBboxPatch(
            (vx - VIEW_BOX_W / 2, y6 - VIEW_BOX_H / 2),
            VIEW_BOX_W, VIEW_BOX_H,
            boxstyle="round,pad=0.08",
            facecolor=C_ORANGE2, edgecolor="white", linewidth=1.0, zorder=3,
        )
        ax.add_patch(box)
        ax.text(vx, y6, "K-Means", ha="center", va="center",
                fontsize=9, fontweight="bold", color=TEXT_WHITE, zorder=4)

    ax.text(MID, y6 + VIEW_BOX_H / 2 + 0.32,
            "6  PCA + K-Means Clustering (per view)",
            ha="center", va="center", fontsize=11,
            color=C_ORANGE2, fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=C_ORANGE2, linewidth=1.2))

    # ── Fan-in ────────────────────────────────────────────────────────
    for vx in view_xs:
        _arrow(ax, vx, y6 - VIEW_BOX_H / 2, MID, y_conv + 0.05,
               color="#888888")

    ax.text(MID, y_conv, "cluster IDs + centroid distances",
            ha="center", va="center", fontsize=9.5, color="#64748B",
            style="italic", zorder=5)

    # ── Arrow to scoring ──────────────────────────────────────────────
    _arrow(ax, MID, y_conv - 0.15, MID, y7 + SMALL_BOX_H / 2 + 0.12)

    # ==================================================================
    # Stage 7 — Scoring (3 parallel scorers)
    # ==================================================================
    s7_label_y = y7 + SMALL_BOX_H / 2 + 0.5
    ax.text(MID, s7_label_y,
            "7  Quality Scoring (3 independent scorers)",
            ha="center", va="center", fontsize=12,
            color=C_GREEN1, fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=C_GREEN1, linewidth=1.2))

    score_xs = [MID - 3.6, MID, MID + 3.6]
    score_colors = [C_GREEN1, C_GREEN2, C_GREEN3]
    score_labels = [
        ("7a  Affiliation", "QS rankings + company tiers"),
        ("7b  Citation", "OpenAlex + Semantic Scholar"),
        ("7c  H-Index", "max author h-index"),
    ]

    for sx, sc, (sl_top, sl_bot) in zip(score_xs, score_colors, score_labels):
        _arrow(ax, MID, s7_label_y - 0.3, sx, y7 + SMALL_BOX_H / 2, color=sc)
        _box(ax, sx, y7, SMALL_BOX_W, SMALL_BOX_H, sc,
             sl_top, sl_bot, fs_top=10, fs_bot=8)

    # ── Arrows to output ──────────────────────────────────────────────
    for sx in score_xs:
        _arrow(ax, sx, y7 - SMALL_BOX_H / 2, MID, y8 + BOX_H / 2,
               color="#888888")

    # ==================================================================
    # Stage 8 — CSV Output
    # ==================================================================
    _box(ax, MID, y8, BOX_W, BOX_H, C_PINK,
         "8  Output", "results.csv + section_diagnostics.csv")

    # ==================================================================
    # Side annotations
    # ==================================================================

    # SQLite Cache (left)
    cache_x = 2.8
    cache_y = (y1 + y6) / 2 + 1.0
    cache_box = FancyBboxPatch(
        (cache_x - 1.6, cache_y - 1.6), 3.2, 3.2,
        boxstyle="round,pad=0.15",
        facecolor="#F1F5F9", edgecolor="#94A3B8", linewidth=1.5,
        linestyle="--", zorder=3,
    )
    ax.add_patch(cache_box)
    ax.text(cache_x, cache_y + 1.0, "SQLite Cache",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="#475569", zorder=4)
    for i, item in enumerate(["pdf_text", "metadata", "sections",
                              "summaries", "embeddings",
                              "citations", "h-index"]):
        ax.text(cache_x, cache_y + 0.5 - i * 0.35, f"- {item}",
                ha="center", va="center", fontsize=8.5,
                color="#64748B", zorder=4)

    ax.annotate("", xy=(MID - BOX_W / 2 - 0.1, y2),
                xytext=(cache_x + 1.6, cache_y),
                arrowprops=dict(arrowstyle="<->", color=C_SIDE,
                                linewidth=1.2, linestyle="--"), zorder=2)
    ax.text(cache_x + 2.5, (cache_y + y2) / 2 + 0.8,
            "SHA256-keyed\nresumability",
            ha="center", va="center", fontsize=8.5,
            color="#64748B", style="italic", zorder=4)

    # External APIs (right, near scoring)
    api_x = FIG_W - 2.8
    api_y = y7 + 0.2
    api_box = FancyBboxPatch(
        (api_x - 1.5, api_y - 0.9), 3.0, 1.8,
        boxstyle="round,pad=0.15",
        facecolor="#F1F5F9", edgecolor="#94A3B8", linewidth=1.5,
        linestyle="--", zorder=3,
    )
    ax.add_patch(api_box)
    ax.text(api_x, api_y + 0.5, "External APIs",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="#475569", zorder=4)
    ax.text(api_x, api_y + 0.05, "- OpenAlex (10 RPS)",
            ha="center", va="center", fontsize=9, color="#64748B", zorder=4)
    ax.text(api_x, api_y - 0.28, "- Semantic Scholar (1 RPS)",
            ha="center", va="center", fontsize=9, color="#64748B", zorder=4)

    ax.annotate("", xy=(score_xs[2] + SMALL_BOX_W / 2 + 0.1, y7),
                xytext=(api_x - 1.5, api_y - 0.2),
                arrowprops=dict(arrowstyle="-|>", color=C_SIDE,
                                linewidth=1.0, linestyle="--"), zorder=2)

    # vLLM Server (right, near stages 3-4)
    vllm_x = FIG_W - 2.8
    vllm_y = (y3 + y4b) / 2
    _side_box(ax, vllm_x, vllm_y,
              "vLLM Server (Qwen3-30B)",
              MID + BOX_W / 2 + 0.1, y3, color="#7C3AED")

    # Embedding model (left, near stage 5)
    _side_box(ax, 2.8, y5,
              "Qwen3-Embedding-4B (GPU)",
              MID - BOX_W / 2 - 0.1, y5, color=C_ORANGE1)

    # ==================================================================
    # Legend
    # ==================================================================
    legend_y = 2.8
    legend_items = [
        (C_BLUE2, "Extraction (1-3)"),
        (C_PURPLE1, "Sections (4a-4b)"),
        (C_ORANGE1, "Embed + Cluster (5-6)"),
        (C_GREEN2, "Scoring (7)"),
        (C_PINK, "Output (8)"),
    ]
    total_w = len(legend_items) * 3.2
    start_x = MID - total_w / 2 + 0.5
    for i, (color, label) in enumerate(legend_items):
        lx = start_x + i * 3.2
        box = FancyBboxPatch(
            (lx - 0.3, legend_y - 0.15), 0.35, 0.3,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="white", linewidth=0.8, zorder=3,
        )
        ax.add_patch(box)
        ax.text(lx + 0.3, legend_y, label, ha="left", va="center",
                fontsize=9, color=TEXT_DARK, zorder=4)

    # ── Save ──────────────────────────────────────────────────────────
    base = Path(__file__).resolve().parent.parent
    docs = base / "docs"
    docs.mkdir(exist_ok=True)

    out_pdf = docs / "pipeline_diagram.pdf"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight",
                dpi=200, facecolor=BG_COLOR)
    print(f"Pipeline diagram saved to {out_pdf}")

    if "--png" in sys.argv:
        out_png = docs / "pipeline_diagram.png"
        fig.savefig(out_png, format="png", bbox_inches="tight",
                    dpi=150, facecolor=BG_COLOR)
        print(f"PNG preview saved to {out_png}")

    plt.close(fig)


if __name__ == "__main__":
    generate()
