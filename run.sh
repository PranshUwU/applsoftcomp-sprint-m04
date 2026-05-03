#!/usr/bin/env bash
# Reproducible pipeline for Sprint M04 — Semantic Axes (S&P 500).
#
# Run: bash run.sh
# Requirements: uv must be installed (https://docs.astral.sh/uv/)
# Everything else is handled automatically via inline script dependencies.

set -euo pipefail

mkdir -p figs

# ---------------------------------------------------------------------------
# Step 1 — Data is already committed in data/sp500.csv.
#           No download step needed.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 2 — Run the analysis and generate the figure.
#           uvx reads the inline [script] dependency block in submission.py
#           and installs everything in an isolated environment automatically.
# ---------------------------------------------------------------------------
echo "Running submission.py …"
uvx --with sentence-transformers \
    --with numpy \
    --with pandas \
    --with matplotlib \
    --with scipy \
    python submission.py

echo "Done. Figure saved to figs/sp500_semantic_map.png"
