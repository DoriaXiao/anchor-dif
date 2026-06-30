#!/usr/bin/env bash
# Reproduce the manuscript end to end.
# Requires: R (>= 4.5) with readr, dplyr, tidyr, ggplot2, knitr; Quarto (>= 1.5); a LaTeX install.
set -euo pipefail
cd "$(dirname "$0")"

echo "Rendering the manuscript (recomputes every reported number from data/constat/) ..."
quarto render paper/anchor_dif.qmd --to apaquarto-pdf

echo "Done. Output: paper/anchor_dif.pdf"
