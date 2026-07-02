# -*- coding: utf-8 -*-
"""
Step 2 (v1): contamination as anchor-invariance violation, at the BENCHMARK-as-anchor level.

Why benchmark-level and not per-item DIF: with only 12 models, per-item DIF (12 obs/item,
8-vs-4 group split) is statistically hopeless -- separable logits, no MH strata. So v1 tests
each benchmark as a candidate anchor set and asks whether it functions invariantly across
"could-have-seen-it" vs "released-after-cutoff" model groups, conditioning on ability.

Method (leave-one-benchmark-out residual / anchor-DIF):
  1. For benchmark B, estimate each model's ability theta^(-B) from a Rasch fit on all OTHER
     benchmarks (the anchor). This places models on a scale that did NOT use B.
  2. Predict each model's expected accuracy on B from theta^(-B) and B's item difficulties.
  3. Residual r_{m,B} = observed_acc - expected_acc  (positive = B is differentially easy for m).
  4. Contamination test: is mean residual higher for EXPOSED models (cutoff >= release(B))
     than CLEAN models (cutoff < release(B))? Effect = mean_exposed - mean_clean.
  5. Purify: drop benchmarks flagged as differentially-easy-for-exposed anchors; re-fit ability
     on the purified anchor set; compare leaderboard ranking before vs after  -> money figure.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from anchordif import (load_responses, fit_rasch_jml, benchmark_boundaries,
                       item_benchmark_index, ROW_CUTOFF)

OUT = os.path.dirname(__file__)
X = load_responses()
M, I = X.shape
bnd = benchmark_boundaries()
bidx = item_benchmark_index()
sig = lambda z: 1.0 / (1.0 + np.exp(-z))


def fit_rasch_subset(Xsub):
    """Rasch on a column subset; return theta[12], b[k], var_mask over the subset."""
    return fit_rasch_jml(Xsub)


def expected_acc(theta, b):
    """Mean P(correct) over items given Rasch theta (per model) and difficulties b."""
    P = sig(theta[:, None] - b[None, :])
    return P.mean(axis=1)


cut = {r: ROW_CUTOFF[r][1] for r in range(M)}            # row -> 'YYYY-MM'
rows_sorted = sorted(range(M), key=lambda r: -X[r].mean())

# ---------- leave-one-benchmark-out residuals + contamination test ----------
print(f"{'benchmark':18s} {'rel':8s} {'exp':>4s} {'cln':>4s} {'r_exp':>7s} {'r_cln':>7s} {'gap':>7s} flag")
report = ["# Step 2 (v1): benchmark-as-anchor contamination test\n"]
report.append("Residual r = observed_acc(B) - expected_acc(B | ability from all OTHER benchmarks).")
report.append("EXPOSED = cutoff >= release(B); CLEAN = cutoff < release(B). gap = mean r_exposed - mean r_clean.\n")
report.append(f"{'benchmark':18s} {'release':8s} {'#exp':>5s} {'#cln':>5s} {'r_exp':>7s} {'r_cln':>7s} {'gap':>7s} flag")

results = []
for name, s, e, rel in bnd:
    others = np.ones(I, dtype=bool); others[s:e] = False
    th, b_other, vm = fit_rasch_subset(X[:, others])     # ability from anchor = other benchmarks
    # difficulties for B's items: fit Rasch difficulty for B given fixed theta
    XB = X[:, s:e]
    vmB = ~((XB.sum(0) == 0) | (XB.sum(0) == M))
    XBv = XB[:, vmB]
    bB = np.zeros(XBv.shape[1])
    for _ in range(100):
        P = sig(th[:, None] - bB[None, :])
        bB -= np.sum(-(XBv - P), axis=0) / -np.sum(P * (1 - P), axis=0)
    obs = XBv.mean(1)
    exp = expected_acc(th, bB)
    resid = obs - exp
    exposed = np.array([cut[r] >= rel for r in range(M)])  # string compare works for YYYY-MM
    r_exp = resid[exposed].mean() if exposed.any() else np.nan
    r_cln = resid[~exposed].mean() if (~exposed).any() else np.nan
    gap = r_exp - r_cln
    flag = "<<< DIF" if (np.isfinite(gap) and gap > 0.05) else ""
    print(f"{name:18s} {rel:8s} {exposed.sum():4d} {(~exposed).sum():4d} "
          f"{r_exp:7.3f} {r_cln:7.3f} {gap:7.3f} {flag}")
    report.append(f"{name:18s} {rel:8s} {exposed.sum():5d} {(~exposed).sum():5d} "
                  f"{r_exp:7.3f} {r_cln:7.3f} {gap:7.3f} {flag}")
    results.append((name, s, e, rel, gap, flag != ""))

flagged = [name for name, *_, fl in results if fl]
report.append(f"\nFlagged (differentially easy for exposed models, gap>0.05): {flagged or 'none'}")

# ---------- ranking shift: full anchor vs purified anchor ----------
theta_full, _, _ = fit_rasch_jml(X)
keep = np.ones(I, dtype=bool)
for name, s, e, rel, gap, fl in results:
    if fl:
        keep[s:e] = False
theta_pure, _, _ = fit_rasch_jml(X[:, keep])

rank_full = (-theta_full).argsort().argsort()    # 0 = best
rank_pure = (-theta_pure).argsort().argsort()
report.append("\n# Leaderboard before vs after purifying flagged anchors")
report.append(f"{'row':>3s} {'model(assumed)':28s} {'rank_full':>9s} {'rank_pure':>9s} {'shift':>6s}")
for r in rows_sorted:
    nm = ROW_CUTOFF[r][0]
    shift = rank_full[r] - rank_pure[r]   # +ve = moved up after purification
    report.append(f"{r:3d} {nm:28s} {rank_full[r]+1:9d} {rank_pure[r]+1:9d} {shift:+6d}")

txt = "\n".join(report)
with open(os.path.join(OUT, "results_contamination.txt"), "w") as f:
    f.write(txt + "\n")
print("\n" + "\n".join(report[-(M+3):]))

# ---------- figures ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# (left) contamination gap per benchmark, ordered by release date
order = sorted(range(len(results)), key=lambda i: results[i][3])
names = [results[i][0] for i in order]
gaps = [results[i][4] for i in order]
cols = ["#C44E52" if results[i][5] else "#9aa0a6" for i in order]
ax[0].barh(range(len(names)), gaps, color=cols)
ax[0].set_yticks(range(len(names)))
ax[0].set_yticklabels([f"{results[i][0]} ({results[i][3][:4]})" for i in order], fontsize=8)
ax[0].axvline(0.05, ls="--", c="k", lw=.8)
ax[0].set_xlabel("residual gap  (exposed - clean)")
ax[0].set_title("Differential easiness for exposed models\n(red = flagged anchor)")

# (right) ranking shift before vs after purification (bump chart)
for r in range(M):
    ax[1].plot([0, 1], [rank_full[r] + 1, rank_pure[r] + 1], "-o", color="#4C72B0", alpha=.7)
    ax[1].annotate(f"r{r}", (1.02, rank_pure[r] + 1), fontsize=7, va="center")
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["full anchor", "purified anchor"])
ax[1].invert_yaxis(); ax[1].set_ylabel("leaderboard rank (1=best)")
ax[1].set_title("Ranking shift after dropping flagged anchors")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_contamination_ranking_shift.png"), dpi=130)
print("\nwrote results_contamination.txt and fig_contamination_ranking_shift.png")
