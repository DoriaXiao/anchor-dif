# -*- coding: utf-8 -*-
"""
anchor-DIF (paper #2) — shared utilities.

Core data: PSN-IRT release. combine.csv is a 12 (models, rows) x 41,871 (items, cols)
binary response matrix, values in {0,1}, no header, no missing.

Established facts (verified 2026-06-29):
  * Columns are concatenated by benchmark in ALPHABETICAL order. Verified by slicing the
    authors' results/item_parameters.csv with the boundaries below and matching the per-
    benchmark means in dataset_analysis.csv to 3 decimals on ALL 11 benchmarks.
  * Item counts sum to 41,871 (matches the paper).
  * Row order (student_id 0..11) is consistent across combine.csv, student_abilities.csv
    and item_parameters.csv (Rasch theta from combine.csv corr 0.94 with author abilities,
    item difficulty corr 0.965 with author 4PL difficulty).

UNRESOLVED: row index -> model NAME mapping. The repo anonymizes models as 0..11 and the
paper gives no per-model scores. Names are known (see MODELS) but not yet aligned to rows.
"""
import os
import numpy as np

REPO = os.path.join(os.path.dirname(__file__), "PSN-IRT")
COMBINE = os.path.join(REPO, "data", "combine.csv")

# (benchmark, n_items, release YYYY-MM) in the alphabetical column order used by the matrix.
BENCHMARKS = [
    ("ARC-C",            295,  "2018-03"),
    ("BBH",             6511,  "2022-10"),
    ("Chinese SimpleQA", 3000, "2024-11"),
    ("GPQA Diamond",     198,  "2023-11"),
    ("GSM8K",           1319,  "2021-10"),
    ("HellaSwag",      10042,  "2019-05"),
    ("HumanEval",        164,  "2021-07"),
    ("MATH",            5000,  "2021-03"),
    ("MBPP",             500,  "2021-08"),
    ("MMLU",           14042,  "2020-09"),
    ("TheoremQA",        800,  "2023-05"),
]

# The 12 models named in the paper (NOT yet aligned to row indices 0..11).
# Frontier/API (expected high ability) then small open (expected low ability).
MODELS = [
    "360GPT2-Pro", "DeepSeek-V3", "Doubao-Pro", "Gemini-1.5",
    "Hunyuan-Turbo", "Moonshot-v1", "Qwen-Plus", "Yi-Lightning",
    "Gemma-2B-it", "Mistral-7B-Instruct-v0.1", "Qwen2.5-3B-Instruct", "Vicuna-7B-v1.3",
]

# ---- ASSUMED row -> (model name, training cutoff) mapping. NOT authoritative. ----
# The 8 frontier rows are identified only as a block (all 2024-era cutoffs, so their
# per-benchmark contamination labels are identical; exact within-block identity does
# not affect the cutoff-vs-release analysis). The 4 small-open rows are identified from
# response-pattern fingerprints (see 02 script notes); these are best-guesses and the
# r4<->r10 (Vicuna vs Gemma) assignment in particular should be checked for sensitivity.
# Cutoffs are approximate "training data cutoff" dates used only to label, per benchmark,
# whether a model could have seen that benchmark (cutoff >= release => possibly exposed).
ROW_CUTOFF = {
    0: ("frontier (2024-era)", "2024-06"),
    1: ("frontier (2024-era)", "2024-06"),
    2: ("frontier (2024-era)", "2024-06"),
    3: ("frontier (2024-era)", "2024-06"),
    5: ("frontier (2024-era)", "2024-06"),
    7: ("frontier (2024-era)", "2024-06"),
    8: ("frontier (2024-era)", "2024-06"),
    11: ("frontier (2024-era)", "2024-06"),
    9:  ("Qwen2.5-3B-Instruct (assumed)",      "2024-09"),
    6:  ("Mistral-7B-Instruct-v0.1 (assumed)", "2023-09"),
    10: ("Gemma-2B-it (assumed)",              "2024-02"),
    4:  ("Vicuna-7B-v1.3 (assumed)",           "2023-03"),
}


def benchmark_boundaries():
    """Return list of (name, start, end, release) with 0-based [start, end) col slices."""
    out, s = [], 0
    for name, n, rel in BENCHMARKS:
        out.append((name, s, s + n, rel))
        s += n
    return out


def item_benchmark_index():
    """Return an int array of length 41,871 giving the benchmark id (0..10) for each column."""
    idx = np.empty(sum(n for _, n, _ in BENCHMARKS), dtype=int)
    for bid, (_, s, e, _) in enumerate(benchmark_boundaries()):
        idx[s:e] = bid
    return idx


def load_responses():
    """Return X (12 x 41871) float array of 0/1 responses."""
    return np.loadtxt(COMBINE, delimiter=",")


def fit_rasch_jml(X, n_iter=200):
    """Joint-MLE Rasch (1PL). Drops no-variance items. Returns (theta[12], b[k], var_mask)."""
    M = X.shape[0]
    var_mask = ~((X.sum(0) == 0) | (X.sum(0) == M))
    Xv = X[:, var_mask]
    theta = np.zeros(M)
    b = np.zeros(Xv.shape[1])
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    for _ in range(n_iter):
        for i in range(M):
            p = sig(theta[i] - b)
            theta[i] -= np.sum(Xv[i] - p) / -np.sum(p * (1 - p))
        P = sig(theta[:, None] - b[None, :])
        b -= np.sum(-(Xv - P), axis=0) / -np.sum(P * (1 - P), axis=0)
        theta -= theta.mean()
    return theta, b, var_mask
