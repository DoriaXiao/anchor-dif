# Contamination Inflates Scores but Rarely Reorders Leaderboards

A ground-truth-calibrated measurement-invariance audit of language model benchmarks.

This repository is the reference implementation and reproducible manuscript for the study.
The paper recasts benchmark contamination as a violation of anchor-item invariance and
measures it through the differential functioning of original versus paraphrased items.
The headline finding: across 47 publicly released models on four benchmarks (ARC, GSM8K,
HellaSwag, MMLU), contamination is largely uniform. It inflates absolute scores without
reordering the leaderboard (rank correlation 0.997 between standard and paraphrase-controlled
scoring), and ranking distortion requires the rare case of differential contamination.

The manuscript is a literate document: every number it reports is computed inline from the
response matrices in `data/`, so the text cannot drift from the analysis.

## Repository structure

```
paper/
  anchor_dif.qmd            canonical manuscript (apaquarto + R); the single source of truth
  refs.bib                  references
  prep_observational.csv    summary values for the observational identifiability check
  anchor_dif.pdf            rendered manuscript
data/
  constat/                  derived per-instance correctness matrices (models x items)
  README.md                 data provenance and license
data-prep/
  20_build_constat.py       rebuilds data/constat/ from the public ConStat outputs
  03_exposure_matrix.py     regenerates figures/fig_exposure_matrix.png (needs PSN-IRT)
  anchordif.py              helper used by the observational scripts
figures/
  fig_exposure_matrix.png   observational exposure matrix (static asset)
reproduce.sh                one-command reproduction
```

## Reproduce the manuscript

Requirements: R (>= 4.5), Quarto (>= 1.5), and a LaTeX install (TinyTeX is enough). The
analysis uses base R plus `readr`, `dplyr`, `tidyr`, `ggplot2`, and `knitr`. The exact
versions used are printed by `sessionInfo()` in the manuscript Appendix.

```bash
quarto render paper/anchor_dif.qmd --to apaquarto-pdf
```

This recomputes every reported quantity from `data/constat/`, runs the leaderboard-distortion
simulation under a fixed seed (seed 1), regenerates the figures, and writes `paper/anchor_dif.pdf`.

## Rebuild the data from source (optional)

The matrices in `data/constat/` are derived from the public ConStat model outputs. To rebuild
them, download `compressed_output.zip` from the ConStat data release, set its path in
`data-prep/20_build_constat.py`, and run the script with Python (numpy, pandas). The
observational exposure matrix additionally requires the PSN-IRT repository; see
`data-prep/03_exposure_matrix.py`.

## Data and licensing

The response matrices in `data/` are derived (binary correctness only, no benchmark item text)
from the ConStat release and are redistributed here for reproducibility with attribution. See
`data/README.md` for provenance. The code and manuscript text in this repository are released
under the MIT License (`LICENSE`).

## Citation

```bibtex
@unpublished{xiao2026contamination,
  author = {Xiao, Xingyao},
  title  = {Contamination Inflates Scores but Rarely Reorders Leaderboards:
            A Ground-Truth-Calibrated Measurement-Invariance Audit of Language Model Benchmarks},
  year   = {2026},
  note   = {Manuscript}
}
```
