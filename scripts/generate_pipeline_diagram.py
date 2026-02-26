#!/usr/bin/env python3
"""Generate a presentation-ready PDF diagram of the multi-view clustering pipeline."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 22, 30          # landscape-ish tall page
BOX_W, BOX_H = 3.4, 1.15      # standard box size
SMALL_BOX_W = 2.6              # scoring sub-boxes
SMALL_BOX_H = 1.05
VIEW_BOX_W, VIEW_BOX_H = 1.9, 0.72
ARROW_KW = dict(
    arrowstyle="-|>",
    color="#555555",
    linewidth=1.8,
    mutation_scale=18,
    connectionstyle="arc3,rad=0",
)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
C_BLUE1 = "#3B82F6"   # discovery
C_BLUE2 = "#2563EB"   # full-text
C_BLUE3 = "#1D4ED8"   # metadata
C_PURPLE1 = "#8B5CF6"  # section splitting
C_PURPLE2 = "#7C3AED"  # section summarization
C_ORANGE1 = "#F59E0B"  # embedding
C_ORANGE2 = "#D97706"  # clustering
C_GREEN1 = "#10B981"   # affiliation
C_GREEN2 = "#059669"   # citation
C_GREEN3 = "#047857"   # h-index
C_RED1 = "#EF4444"     # local scores
C_RED2 = "#DC2626"     # fusion
C_PINK = "#EC4899"     # output
C_SIDE = "#94A3B8"     # side annotations
C_VIEW = "#6366F1"     # view fan-out boxes
TEXT_WHITE = "#FFFFFF"
TEXT_DARK = "#1E293B"
BG_COLOR = "#F8FAFC"


def _rounded_box(ax, cx, cy, w, h, color, label_top, label_bot,
                 text_color=TEXT_WHITE, fontsize_top=11, fontsize_bot=9,
                 alpha=1.0, boxstyle="round,pad=0.15"):
    """Draw a rounded rectangle centred at (cx, cy) with two text lines."""
    x0, y0 = cx - w / 2, cy - h / 2
    box = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle=boxstyle,
        facecolor=color, edgecolor="white", linewidth=1.4, alpha=alpha,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy + 0.16, label_top,
            ha="center", va="center", fontsize=fontsize_top,
            fontweight="bold", color=text_color, zorder=4)
    if label_bot:
        ax.text(cx, cy - 0.18, label_bot,
                ha="center", va="center", fontsize=fontsize_bot,
                color=text_color, style="italic", zorder=4, alpha=0.92)
    return box


def _arrow(ax, x1, y1, x2, y2, label=None, color="#555555", rad=0.0):
    """Draw an arrow from (x1,y1) to (x2,y2) with optional label."""
    style = f"arc3,rad={rad}" if rad else "arc3,rad=0"
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", color=color, linewidth=1.6,
        mutation_scale=16, connectionstyle=style, zorder=2,
    )
    ax.add_patch(arr)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # offset label to the right of vertical arrows
        offset_x = 0.15 if abs(x2 - x1) < 0.5 else 0.0
        ax.text(mx + offset_x, my + 0.08, label,
                ha="left", va="center", fontsize=8.5,
                color="#64748B", style="italic", zorder=5)


def _side_annotation(ax, cx, cy, text, anchor_x, anchor_y, color=C_SIDE):
    """Draw a side box with a dashed line pointing to the pipeline."""
    box = FancyBboxPatch(
        (cx - 1.5, cy - 0.38), 3.0, 0.76,
        boxstyle="round,pad=0.12",
        facecolor="#F1F5F9", edgecolor=color, linewidth=1.3,
        linestyle="--", zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=9, color="#475569", zorder=4)
    ax.annotate(
        "", xy=(anchor_x, anchor_y), xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        linewidth=1.0, linestyle="--"),
        zorder=2,
    )


def generate():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # ── Title ──────────────────────────────────────────────────────────
    ax.text(FIG_W / 2, FIG_H - 0.9,
            "Multi-View Clustering Pipeline for Agentic AI Survey",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color=TEXT_DARK, zorder=5)
    ax.text(FIG_W / 2, FIG_H - 1.55,
            "10-Stage Paper Selection System  \u2022  5 Section Views  \u2022  Consensus Fusion",
            ha="center", va="center", fontsize=13, color="#64748B", zorder=5)

    # ── Centre column x positions ──────────────────────────────────────
    MID = FIG_W / 2  # 11

    # ── Y positions (top to bottom) ────────────────────────────────────
    y_s1 = 26.5
    y_s2 = 24.5
    y_s3 = 22.5
    y_s4a = 20.3
    y_s4b = 18.3
    y_filter = 16.7
    y_s5 = 14.8
    y_views = 13.0   # fan-out row
    y_s6 = 11.2      # clustering row
    y_converge = 9.8  # fan-in label
    y_s7 = 8.2       # scoring (3 boxes side by side)
    y_s8 = 6.0
    y_s9 = 4.0
    y_s10 = 2.0

    # ======================================================================
    # STAGE 1 — Discovery
    # ======================================================================
    _rounded_box(ax, MID, y_s1, BOX_W, BOX_H, C_BLUE1,
                 "\u2460  Discovery", "Scan papers/<Venue>/*.pdf")
    _arrow(ax, MID, y_s1 - BOX_H / 2, MID, y_s2 + BOX_H / 2,
           label="Paper objects")

    # ======================================================================
    # STAGE 2 — Full-Text Extraction
    # ======================================================================
    _rounded_box(ax, MID, y_s2, BOX_W, BOX_H, C_BLUE2,
                 "\u2461  Full-Text Extraction", "PyMuPDF (fitz)")
    _arrow(ax, MID, y_s2 - BOX_H / 2, MID, y_s3 + BOX_H / 2,
           label="raw text")

    # ======================================================================
    # STAGE 3 — Metadata Extraction
    # ======================================================================
    _rounded_box(ax, MID, y_s3, BOX_W, BOX_H, C_BLUE3,
                 "\u2462  Metadata Extraction", "Qwen3-30B via vLLM")
    _arrow(ax, MID, y_s3 - BOX_H / 2, MID, y_s4a + BOX_H / 2,
           label="title, abstract, authors, year")

    # ======================================================================
    # STAGE 4a — Section Splitting
    # ======================================================================
    _rounded_box(ax, MID, y_s4a, BOX_W, BOX_H, C_PURPLE1,
                 "\u2463a  Section Splitting", "Regex + LLM fallback")
    _arrow(ax, MID, y_s4a - BOX_H / 2, MID, y_s4b + BOX_H / 2,
           label="5 raw sections")

    # ======================================================================
    # STAGE 4b — Section Summarization
    # ======================================================================
    _rounded_box(ax, MID, y_s4b, BOX_W, BOX_H, C_PURPLE2,
                 "\u2463b  Summarization", "~400-word summaries (vLLM)")
    _arrow(ax, MID, y_s4b - BOX_H / 2, MID, y_filter + 0.28,
           label="section summaries")

    # ── Filter incomplete papers (small node) ─────────────────────────
    ax.add_patch(FancyBboxPatch(
        (MID - 1.4, y_filter - 0.25), 2.8, 0.5,
        boxstyle="round,pad=0.10", facecolor="#FEF3C7",
        edgecolor="#F59E0B", linewidth=1.2, zorder=3,
    ))
    ax.text(MID, y_filter, "Filter incomplete papers",
            ha="center", va="center", fontsize=9,
            color="#92400E", fontweight="bold", zorder=4)
    _arrow(ax, MID, y_filter - 0.28, MID, y_s5 + BOX_H / 2,
           label="complete papers only")

    # ======================================================================
    # STAGE 5 — Multi-View Embedding
    # ======================================================================
    _rounded_box(ax, MID, y_s5, BOX_W, BOX_H, C_ORANGE1,
                 "\u2464  Multi-View Embedding", "Qwen3-Embedding-4B",
                 text_color=TEXT_DARK)

    # ── Fan-out to 5 views ─────────────────────────────────────────────
    view_names = [
        "Title+Abs+Concl",
        "Introduction",
        "Related Work",
        "Method",
        "Experiments",
    ]
    view_xs = [MID - 6.4, MID - 3.2, MID, MID + 3.2, MID + 6.4]

    for i, (vx, vname) in enumerate(zip(view_xs, view_names)):
        # Arrow from embedding box to view box
        _arrow(ax, MID, y_s5 - BOX_H / 2, vx, y_views + VIEW_BOX_H / 2 + 0.05,
               color=C_VIEW)
        # View box
        box = FancyBboxPatch(
            (vx - VIEW_BOX_W / 2, y_views - VIEW_BOX_H / 2),
            VIEW_BOX_W, VIEW_BOX_H,
            boxstyle="round,pad=0.08",
            facecolor=C_VIEW, edgecolor="white", linewidth=1.0, zorder=3,
        )
        ax.add_patch(box)
        ax.text(vx, y_views + 0.06, f"V{i}",
                ha="center", va="center", fontsize=7,
                color="#C7D2FE", zorder=4)
        ax.text(vx, y_views - 0.14, vname,
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color=TEXT_WHITE, zorder=4)

    # ── "5 Section Views" label ────────────────────────────────────────
    ax.text(MID, y_views + VIEW_BOX_H / 2 + 0.42,
            "5 Independent Section Views",
            ha="center", va="center", fontsize=11,
            color=C_VIEW, fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=C_VIEW, linewidth=1.2))

    # ======================================================================
    # STAGE 6 — K-Means Clustering (per view)
    # ======================================================================
    for vx in view_xs:
        _arrow(ax, vx, y_views - VIEW_BOX_H / 2, vx, y_s6 + VIEW_BOX_H / 2 + 0.05,
               color=C_ORANGE2)
        box = FancyBboxPatch(
            (vx - VIEW_BOX_W / 2, y_s6 - VIEW_BOX_H / 2),
            VIEW_BOX_W, VIEW_BOX_H,
            boxstyle="round,pad=0.08",
            facecolor=C_ORANGE2, edgecolor="white", linewidth=1.0, zorder=3,
        )
        ax.add_patch(box)
        ax.text(vx, y_s6, "K-Means",
                ha="center", va="center", fontsize=9,
                fontweight="bold", color=TEXT_WHITE, zorder=4)

    ax.text(MID, y_s6 + VIEW_BOX_H / 2 + 0.35,
            "\u2465  K-Means Clustering (PCA + weak-member detection)",
            ha="center", va="center", fontsize=11,
            color=C_ORANGE2, fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=C_ORANGE2, linewidth=1.2))

    # ── Fan-in: converge arrows ────────────────────────────────────────
    for vx in view_xs:
        _arrow(ax, vx, y_s6 - VIEW_BOX_H / 2, MID, y_converge + 0.05,
               color="#888888")

    ax.text(MID, y_converge, "\u25BC  converge per-view results",
            ha="center", va="center", fontsize=9.5, color="#64748B",
            style="italic", zorder=5)

    # ── Arrow to scoring ───────────────────────────────────────────────
    _arrow(ax, MID, y_converge - 0.2, MID, y_s7 + SMALL_BOX_H / 2 + 0.15)

    # ======================================================================
    # STAGE 7 — Scoring (3 parallel sub-stages)
    # ======================================================================
    s7_label_y = y_s7 + SMALL_BOX_H / 2 + 0.55
    ax.text(MID, s7_label_y,
            "\u2466  Quality Scoring (3 independent scorers)",
            ha="center", va="center", fontsize=12,
            color=C_GREEN1, fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=C_GREEN1, linewidth=1.2))

    score_xs = [MID - 3.8, MID, MID + 3.8]
    score_colors = [C_GREEN1, C_GREEN2, C_GREEN3]
    score_labels = [
        ("7a  Affiliation", "QS rankings + company tiers"),
        ("7b  Citation", "actual / expected for age"),
        ("7c  H-Index", "max author h-index / 50"),
    ]

    for sx, sc, (sl_top, sl_bot) in zip(score_xs, score_colors, score_labels):
        # Arrow from label area to box
        _arrow(ax, MID, s7_label_y - 0.35, sx, y_s7 + SMALL_BOX_H / 2,
               color=sc)
        _rounded_box(ax, sx, y_s7, SMALL_BOX_W, SMALL_BOX_H, sc,
                     sl_top, sl_bot, fontsize_top=10, fontsize_bot=8)

    # ── Arrows from scoring to stage 8 ─────────────────────────────────
    for sx in score_xs:
        _arrow(ax, sx, y_s7 - SMALL_BOX_H / 2, MID, y_s8 + BOX_H / 2,
               color="#888888")

    # ======================================================================
    # STAGE 8 — Within-Cluster Ranking
    # ======================================================================
    _rounded_box(ax, MID, y_s8, BOX_W + 0.6, BOX_H, C_RED1,
                 "\u2467  Within-Cluster Ranking",
                 "h-index 0.35 + affiliation 0.35 + citation 0.30")
    _arrow(ax, MID, y_s8 - BOX_H / 2, MID, y_s9 + BOX_H / 2,
           label="per-view local scores")

    # ======================================================================
    # STAGE 9 — Multi-Criteria Fusion
    # ======================================================================
    _rounded_box(ax, MID, y_s9, BOX_W + 0.6, BOX_H, C_RED2,
                 "\u2468  Multi-Criteria Fusion",
                 "Consensus-weighted across 5 views")
    _arrow(ax, MID, y_s9 - BOX_H / 2, MID, y_s10 + BOX_H / 2,
           label="accept / reject decisions")

    # ======================================================================
    # STAGE 10 — Output
    # ======================================================================
    _rounded_box(ax, MID, y_s10, BOX_W, BOX_H, C_PINK,
                 "\u2469  Output",
                 "CSV report + organized PDFs")

    # ======================================================================
    # Side annotations
    # ======================================================================
    # SQLite Cache (left side, spanning stages 1-6)
    cache_x = 3.0
    cache_y = (y_s1 + y_s6) / 2 + 1.5
    cache_box = FancyBboxPatch(
        (cache_x - 1.8, cache_y - 1.8), 3.6, 3.6,
        boxstyle="round,pad=0.15",
        facecolor="#F1F5F9", edgecolor="#94A3B8", linewidth=1.5,
        linestyle="--", zorder=3,
    )
    ax.add_patch(cache_box)
    ax.text(cache_x, cache_y + 1.1, "SQLite Cache",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="#475569", zorder=4)
    cache_items = [
        "pdf_text", "metadata", "sections",
        "section_summaries", "embeddings",
        "citations", "author_hindex",
    ]
    for i, item in enumerate(cache_items):
        ax.text(cache_x, cache_y + 0.55 - i * 0.38, f"\u2022 {item}",
                ha="center", va="center", fontsize=8.5,
                color="#64748B", zorder=4)

    # Dashed connector from cache to pipeline
    ax.annotate(
        "", xy=(MID - BOX_W / 2 - 0.1, y_s2),
        xytext=(cache_x + 1.8, cache_y),
        arrowprops=dict(arrowstyle="<->", color=C_SIDE,
                        linewidth=1.2, linestyle="--"),
        zorder=2,
    )
    ax.text(cache_x + 2.8, (cache_y + y_s2) / 2 + 1.0,
            "SHA256-keyed\nresumability",
            ha="center", va="center", fontsize=8.5,
            color="#64748B", style="italic", zorder=4)

    # External APIs (right side, near scoring)
    api_x = FIG_W - 3.2
    api_y = y_s7 + 0.3
    api_box = FancyBboxPatch(
        (api_x - 1.8, api_y - 1.1), 3.6, 2.2,
        boxstyle="round,pad=0.15",
        facecolor="#F1F5F9", edgecolor="#94A3B8", linewidth=1.5,
        linestyle="--", zorder=3,
    )
    ax.add_patch(api_box)
    ax.text(api_x, api_y + 0.6, "External APIs",
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="#475569", zorder=4)
    ax.text(api_x, api_y + 0.1, "\u2022 OpenAlex (10 RPS)",
            ha="center", va="center", fontsize=9, color="#64748B", zorder=4)
    ax.text(api_x, api_y - 0.25, "\u2022 Semantic Scholar (1 RPS)",
            ha="center", va="center", fontsize=9, color="#64748B", zorder=4)
    ax.text(api_x, api_y - 0.6, "citations + h-index lookup",
            ha="center", va="center", fontsize=8.5,
            color="#64748B", style="italic", zorder=4)

    # Dashed connector from APIs to scoring boxes
    ax.annotate(
        "", xy=(score_xs[2] + SMALL_BOX_W / 2 + 0.1, y_s7),
        xytext=(api_x - 1.8, api_y - 0.3),
        arrowprops=dict(arrowstyle="-|>", color=C_SIDE,
                        linewidth=1.2, linestyle="--"),
        zorder=2,
    )

    # vLLM Server (right side, near stages 3-4)
    vllm_x = FIG_W - 3.2
    vllm_y = (y_s3 + y_s4b) / 2
    _side_annotation(ax, vllm_x, vllm_y,
                     "vLLM Server (Qwen3-30B, GPU)",
                     MID + BOX_W / 2 + 0.1, y_s3, color="#7C3AED")

    # Embedding model (left side, near stage 5)
    emb_x = 3.0
    emb_y = y_s5
    _side_annotation(ax, emb_x, emb_y,
                     "Qwen3-Embedding-4B (CUDA)",
                     MID - BOX_W / 2 - 0.1, y_s5, color=C_ORANGE1)

    # ======================================================================
    # Legend
    # ======================================================================
    legend_y = 0.7
    legend_items = [
        (C_BLUE2, "Extraction (1-3)"),
        (C_PURPLE1, "Section Processing (4a-4b)"),
        (C_ORANGE1, "Embedding + Clustering (5-6)"),
        (C_GREEN2, "Quality Scoring (7)"),
        (C_RED1, "Ranking + Fusion (8-9)"),
        (C_PINK, "Output (10)"),
    ]
    legend_start_x = MID - 3.0 * len(legend_items) / 2
    for i, (color, label) in enumerate(legend_items):
        lx = legend_start_x + i * 3.2
        box = FancyBboxPatch(
            (lx - 0.3, legend_y - 0.15), 0.35, 0.3,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="white", linewidth=0.8, zorder=3,
        )
        ax.add_patch(box)
        ax.text(lx + 0.3, legend_y, label,
                ha="left", va="center", fontsize=9, color=TEXT_DARK, zorder=4)

    # ── Save ───────────────────────────────────────────────────────────
    base = Path(__file__).resolve().parent.parent
    out_pdf = base / "docs" / "pipeline_diagram.pdf"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight",
                dpi=200, facecolor=BG_COLOR)
    print(f"Pipeline diagram saved to {out_pdf}")

    if "--png" in sys.argv:
        out_png = base / "docs" / "pipeline_diagram.png"
        fig.savefig(out_png, format="png", bbox_inches="tight",
                    dpi=150, facecolor=BG_COLOR)
        print(f"PNG preview saved to {out_png}")

    plt.close(fig)


if __name__ == "__main__":
    import sys
    generate()
