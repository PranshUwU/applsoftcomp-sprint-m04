# -*- coding: utf-8 -*-
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers>=2.7.0",
#     "numpy>=1.24",
#     "pandas>=2.0",
#     "matplotlib>=3.7",
#     "scipy>=1.11",
# ]
# ///
"""
Sprint M04 — Semantic Axes: S&P 500 Companies
==============================================
Scores every S&P 500 company along two semantic axes and produces
a publication-quality 2D scatterplot saved to figs/sp500_semantic_map.png.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — works on CI / grader machines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# ---------------------------------------------------------------------------
# 0.  Setup
# ---------------------------------------------------------------------------
os.makedirs("figs", exist_ok=True)

print("Loading embedding model …")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model ready.\n")


# ---------------------------------------------------------------------------
# 1.  Helper functions  (copied from assignment.py reference implementation)
# ---------------------------------------------------------------------------

def make_axis(positive_words, negative_words, embedding_model):
    """Return a unit-length semantic axis from two word sets."""
    pos_emb = embedding_model.encode(positive_words, normalize_embeddings=True)
    neg_emb = embedding_model.encode(negative_words, normalize_embeddings=True)
    pole_pos = pos_emb.mean(axis=0)
    pole_neg = neg_emb.mean(axis=0)
    v = pole_pos - pole_neg
    return v / (np.linalg.norm(v) + 1e-10)


def score_words(words, axis, embedding_model):
    """Project each word onto the axis. Returns one score per word."""
    emb = embedding_model.encode(list(words), normalize_embeddings=True)
    return emb @ axis


def pole_distance(pos_words, neg_words, embedding_model):
    """Cosine distance between the two pole centroids (should be >= 0.3)."""
    pos_emb = embedding_model.encode(pos_words, normalize_embeddings=True)
    neg_emb = embedding_model.encode(neg_words, normalize_embeddings=True)
    return cosine(pos_emb.mean(axis=0), neg_emb.mean(axis=0))


# ---------------------------------------------------------------------------
# 2.  Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(
    "data/sp500.csv",
    dtype={"name": "string", "sector": "category"},
).dropna(subset=["name", "sector"])

print(f"Loaded {len(df)} companies across {df['sector'].nunique()} sectors.\n")


# ---------------------------------------------------------------------------
# 3.  Define the two semantic axes
# ---------------------------------------------------------------------------

# --- Axis 1 (X): Old Economy  ↔  New Economy / Tech ---
axis1_pos = [
    "software startup",
    "digital technology company",
    "Silicon Valley innovation",
    "cloud computing platform",
    "artificial intelligence",
    "disruptive tech company",
]
axis1_neg = [
    "oil and gas drilling",
    "heavy manufacturing plant",
    "railroad freight transport",
    "traditional brick and mortar",
    "legacy industrial company",
    "fossil fuel extraction",
]

# --- Axis 2 (Y): Defensive / Stable  ↔  Speculative / High-Growth ---
axis2_pos = [
    "high risk high reward investment",
    "volatile growth stock",
    "speculative venture",
    "moonshot aggressive expansion",
    "disruptive challenger brand",
    "hypergrowth startup",
]
axis2_neg = [
    "safe dividend income stock",
    "stable utility provider",
    "recession proof consumer necessity",
    "low volatility blue chip",
    "steady predictable cash flow",
    "defensive income investment",
]

# Validate pole separation
d1 = pole_distance(axis1_pos, axis1_neg, model)
d2 = pole_distance(axis2_pos, axis2_neg, model)
print(f"Axis 1 pole distance: {d1:.3f}  {'ok' if d1 >= 0.3 else 'FAIL (< 0.3)'}")
print(f"Axis 2 pole distance: {d2:.3f}  {'ok' if d2 >= 0.3 else 'FAIL (< 0.3)'}\n")

# Build axes
axis_tech   = make_axis(axis1_pos, axis1_neg, model)
axis_growth = make_axis(axis2_pos, axis2_neg, model)


# ---------------------------------------------------------------------------
# 4.  Score every company
# ---------------------------------------------------------------------------
print("Scoring companies …")
x = score_words(df["name"].tolist(), axis_tech,   model)
y = score_words(df["name"].tolist(), axis_growth, model)
df = df.assign(x=x, y=y)
print("Done.\n")


# ---------------------------------------------------------------------------
# 5.  Plot
# ---------------------------------------------------------------------------

# Okabe–Ito extended palette — colorblind-safe, 11 sectors
SECTOR_STYLES = {
    "Communication Services": {"color": "#0072B2", "marker": "o"},
    "Consumer Discretionary": {"color": "#E69F00", "marker": "s"},
    "Consumer Staples":       {"color": "#56B4E9", "marker": "^"},
    "Energy":                 {"color": "#D55E00", "marker": "D"},
    "Financials":             {"color": "#CC79A7", "marker": "P"},
    "Health Care":            {"color": "#009E73", "marker": "X"},
    "Industrials":            {"color": "#F0E442", "marker": "h"},
    "Information Technology": {"color": "#000000", "marker": "*"},
    "Materials":              {"color": "#8B4513", "marker": "v"},
    "Real Estate":            {"color": "#999999", "marker": "<"},
    "Utilities":              {"color": "#4B0082", "marker": "p"},
}

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# Zero lines — help orient the reader
ax.axhline(0, color="#444c56", linewidth=0.8, linestyle="--", zorder=1)
ax.axvline(0, color="#444c56", linewidth=0.8, linestyle="--", zorder=1)

# Plot each sector separately so the legend is clean
for sector, grp in df.groupby("sector", observed=True):
    style = SECTOR_STYLES.get(str(sector), {"color": "#ffffff", "marker": "o"})
    ax.scatter(
        grp["x"], grp["y"],
        c=style["color"],
        marker=style["marker"],
        s=55,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
        label=str(sector),
    )

# Quadrant annotations
quadrant_kw = dict(fontsize=8.5, color="#6e7681", style="italic",
                   ha="center", va="center", zorder=2)
ax.text( 0.72,  0.93, "New Economy\n+ High Growth",  transform=ax.transAxes, **quadrant_kw)
ax.text( 0.72,  0.07, "New Economy\n+ Defensive",    transform=ax.transAxes, **quadrant_kw)
ax.text( 0.12,  0.93, "Old Economy\n+ High Growth",  transform=ax.transAxes, **quadrant_kw)
ax.text( 0.12,  0.07, "Old Economy\n+ Defensive",    transform=ax.transAxes, **quadrant_kw)

# Label a handful of notable companies
HIGHLIGHTS = {
    "Nvidia":       dict(xytext=(12,  6)),
    "Apple Inc.":   dict(xytext=(-14, 8)),
    "Tesla, Inc.":  dict(xytext=(10, -10)),
    "ExxonMobil":   dict(xytext=(8, -8)),
    "Coca-Cola Company (The)": dict(xytext=(8, 6)),
    "Goldman Sachs": dict(xytext=(8, -8)),
    "NextEra Energy": dict(xytext=(8, 6)),
    "Amazon":       dict(xytext=(8, 6)),
    "Microsoft":    dict(xytext=(8, -10)),
    "Boeing":       dict(xytext=(8, 6)),
}
for _, row in df[df["name"].isin(HIGHLIGHTS)].iterrows():
    kw = HIGHLIGHTS[str(row["name"])]
    display = str(row["name"]).replace(" (The)", "").replace(", Inc.", "").replace(" Inc.", "")
    ax.annotate(
        display,
        xy=(row["x"], row["y"]),
        xytext=kw["xytext"],
        textcoords="offset points",
        fontsize=7.2,
        color="#e6edf3",
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#555", lw=0.6),
        zorder=5,
    )

# Legend
legend = ax.legend(
    title="Sector",
    title_fontsize=9,
    fontsize=8,
    loc="lower right",
    framealpha=0.15,
    facecolor="#161b22",
    edgecolor="#30363d",
    labelcolor="#e6edf3",
)
legend.get_title().set_color("#e6edf3")

# Axis labels, title
ax.set_xlabel(
    "← Old Economy / Industrial          New Economy / Tech →",
    fontsize=11, color="#e6edf3", labelpad=10,
)
ax.set_ylabel(
    "← Defensive / Stable          Speculative / High-Growth →",
    fontsize=11, color="#e6edf3", labelpad=10,
)
ax.set_title(
    "S&P 500 Companies in Semantic Space",
    fontsize=15, fontweight="bold", color="#e6edf3", pad=14,
)
ax.tick_params(colors="#6e7681", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")

fig.tight_layout()
out_path = "figs/sp500_semantic_map.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Figure saved -> {out_path}")
