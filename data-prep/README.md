# Data preparation

These scripts rebuild the inputs consumed by the manuscript. They are provided for provenance
and are not needed to reproduce the paper from the shipped inputs (`quarto render paper/anchor_dif.qmd`
uses `../data/constat/` and `../paper/prep_observational.csv`, both included in the repository).

## Primary analysis inputs (Python: numpy, pandas)

- `20_build_constat.py` rebuilds `data/constat/*.csv` (the model-by-item correctness matrices for
  the normal, rephrase, and synthetic versions of ARC, GSM8K, HellaSwag, MMLU) from the public
  ConStat model-output release (`compressed_output.zip`). Set the zip path at the top of the script.
  This is the only external download the main results depend on.

## Observational identifiability check (Section: "Observational grouping ... is not identified")

The numbers in `paper/prep_observational.csv` (the PSN-IRT and HELM figures: 12-model counts,
the HELM Classic sign-instability coefficients, and the ability-balanced Omni-MATH effect) are
produced by the scripts below. They require additional external downloads and are therefore not
run by the default reproduction; `prep_observational.csv` ships the resulting values.

- `03_exposure_matrix.py` builds `figures/fig_exposure_matrix.png` and the 12-model
  identifiability counts. Requires the PSN-IRT repository (`combine.csv`); uses `anchordif.py`.
- `02_contamination_dif.py` the 12-model exposure/DIF analysis. Requires PSN-IRT.
- `10_build_helm_matrix.py`, `11_helm_contamination_dif.py`, `12_build_capabilities.py` build the
  69-model HELM Classic and 68-model HELM Capabilities matrices from the public HELM GCS bucket
  (`crfm-helm-public`) and run the temporal-grouping and ability-balanced contrasts. These
  produce the HELM coefficients reported in `prep_observational.csv`.

## Note

The paraphrased and synthetic benchmark item text in the ConStat release is password protected
to limit recontamination and is not redistributed. Only binary correctness is used here; see
`../data/README.md`.
