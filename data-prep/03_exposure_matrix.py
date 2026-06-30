# -*- coding: utf-8 -*-
"""Diagnostic: model x benchmark EXPOSURE matrix (cutoff >= release?).

This visualizes WHY the cutoff-vs-release contamination DIF cannot be identified on the
12-model PSN-IRT set: a DIF/contamination test needs, for each anchor benchmark, BOTH an
exposed group and a clean (released-after-cutoff) reference group. Here almost every cell
is exposed, so 9/11 benchmarks have no clean reference at all.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False  # ASCII hyphen-minus in figure labels
import matplotlib.pyplot as plt
from anchordif import load_responses, benchmark_boundaries, ROW_CUTOFF

OUT = os.path.dirname(__file__)
X = load_responses()
M = X.shape[0]
bnd = benchmark_boundaries()
rows_sorted = sorted(range(M), key=lambda r: -X[r].mean())   # strongest first

E = np.zeros((M, len(bnd)))   # 1 = exposed (cutoff >= release), 0 = clean
for ri, r in enumerate(rows_sorted):
    cutoff = ROW_CUTOFF[r][1]
    for bi, (_, _, _, rel) in enumerate(bnd):
        E[ri, bi] = 1.0 if cutoff >= rel else 0.0

n_clean_per_bench = (E == 0).sum(0)
n_exposed_per_bench = (E == 1).sum(0)

fig, ax = plt.subplots(figsize=(11, 6))
ax.imshow(E, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
ax.set_xticks(range(len(bnd)))
ax.set_xticklabels([f"{n}\n({rel[:4]})\nclean={int(c)}"
                    for (n, _, _, rel), c in zip(bnd, n_clean_per_bench)],
                   fontsize=7, rotation=45, ha="right")
ax.set_yticks(range(M))
ax.set_yticklabels([f"r{r} · {ROW_CUTOFF[r][0].split(' (')[0]} · {ROW_CUTOFF[r][1]}"
                    for r in rows_sorted], fontsize=7)
for ri in range(M):
    for bi in range(len(bnd)):
        ax.text(bi, ri, "E" if E[ri, bi] else "C", ha="center", va="center",
                fontsize=6, color="white" if E[ri, bi] else "black")
ax.set_title("Exposure matrix: E=cutoff>=release (possibly contaminated), C=clean reference\n"
             "Contamination DIF is unidentified where a benchmark has 0 clean models")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_exposure_matrix.png"), dpi=130)

print("clean models available per benchmark (need >=1 in BOTH groups to test DIF):")
for (n, _, _, rel), c, ex in zip(bnd, n_clean_per_bench, n_exposed_per_bench):
    ok = "testable" if (c >= 2 and ex >= 2) else "NOT identifiable"
    print(f"  {n:18s} rel={rel}  exposed={int(ex):2d} clean={int(c):2d}  -> {ok}")
print("\nwrote fig_exposure_matrix.png")
