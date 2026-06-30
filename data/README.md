# Data provenance

The files in `constat/` are derived from the public model-output release accompanying ConStat
(Dekoninck, Mueller, and Vechev, 2024; arXiv:2405.16281). The original release provides, for
each model and benchmark, per-instance evaluation logs for several benchmark versions.

## What is here

For each benchmark in {arc, gsm8k, hellaswag, mmlu} and each version in {normal, rephrase,
synthetic}, one CSV named `<benchmark>_<version>.csv`:

- Rows are models (the first column is the model id), columns are item identifiers (doc_id).
- Each cell is binary correctness (1 correct, 0 incorrect), or empty if the model has no
  response for that item. The correctness metric is exact_match for gsm8k, acc for mmlu, and
  acc_norm for arc and hellaswag.
- `normal` and `rephrase` share item identifiers and are aligned one to one (original wording
  versus a semantically equivalent paraphrase). `synthetic` is a separate set of newly
  generated equivalent items.

`models_meta.csv` lists every model with its group (real or controlled), and for controlled
models the base model, target benchmark, and contamination condition.

These matrices contain only binary correctness, not the benchmark item text. The paraphrased
and synthetic item text in the ConStat release is password protected to limit recontamination
and is not redistributed here.

## How to rebuild

`../data-prep/20_build_constat.py` reconstructs every CSV from the ConStat `compressed_output.zip`.
Item identifiers and the correctness metric per benchmark are documented in that script.

## License

Derived data redistributed for reproducibility with attribution to the ConStat authors. Refer
to the ConStat repository for the terms governing the original outputs.
