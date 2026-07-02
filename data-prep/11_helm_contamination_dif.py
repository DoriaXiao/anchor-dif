# -*- coding: utf-8 -*-
"""
Real anchor-DIF on the scaled-up HELM matrix (~67 models, cutoff spread 2019-2023).

Now that each benchmark has BOTH a clean (released-after-model) and an exposed group with
many members, per-item DIF is identifiable. For each benchmark B (release R):
  exposed = model release_date >= R ; clean = release_date < R.
We test each item for UNIFORM DIF easier-for-exposed (the contamination signature) via
logistic-regression DIF (Swaminathan-Rogers), matching on ability estimated from the OTHER
benchmarks (leave-one-benchmark-out anchor, avoids circularity):
      logit P(correct) = b0 + b1*theta_anchor + b2*exposed
b2 > 0 (significant + effect size) => item is differentially easy for possibly-contaminated
models given ability => flagged contaminated anchor. Then purify (drop flagged) and compare
the IRT leaderboard before vs after -> the money figure.
"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DATADIR = sys.argv[1] if len(sys.argv) > 1 else "data_helm"   # e.g. data_helm_cap
TAG = "" if DATADIR == "data_helm" else "_" + DATADIR.replace("data_helm_", "")
D = os.path.join(os.path.dirname(__file__), DATADIR)
OUT = os.path.dirname(__file__)
resp = pd.read_csv(os.path.join(D, "helm_responses.csv"), index_col=0)
items = pd.read_csv(os.path.join(D, "helm_items.csv"))
mods = pd.read_csv(os.path.join(D, "helm_models.csv"))
X = resp.values.astype(float)                       # models x items, NaN allowed
M, I = X.shape
bench = items.set_index("item_id").loc[resp.columns, "benchmark"].values
release = items.set_index("item_id").loc[resp.columns, "release"].values
mrel = mods.set_index("model").loc[resp.index, "release_date"].fillna("9999-99").values
mrel = np.array([str(s)[:7] for s in mrel])
sig = lambda z: 1.0/(1.0+np.exp(-z))


def rasch_theta(Xsub, lam=0.1):
    """Penalized Rasch JML (N(0,1/lam) shrinkage prior on theta & b) with NaN support.
    The prior prevents theta -> +/-inf for models that are (near-)perfect/zero on the anchor."""
    obs = np.isfinite(Xsub)
    keep = (np.nansum(Xsub, 0) > 0) & (np.nansum(1-Xsub, 0) > 0)  # variance among observed
    Xs, ob = Xsub[:, keep], obs[:, keep]
    Xz = np.where(ob, Xs, 0.0)
    theta = np.zeros(M); b = np.zeros(Xs.shape[1])
    for _ in range(150):
        for i in range(M):
            p = sig(theta[i]-b)
            g = np.sum((Xz[i]-p)*ob[i]) - lam*theta[i]
            h = -np.sum(p*(1-p)*ob[i]) - lam
            theta[i] -= g/h
        P = sig(theta[:,None]-b[None,:])
        g = np.sum(-(Xz-P)*ob, 0) - lam*b; h = -np.sum(P*(1-P)*ob, 0) - lam
        b -= g/h; theta -= np.nanmean(theta)
    return theta


def lr_dif(y, theta, grp, ridge=1e-3):
    """IRLS logistic DIF: y ~ 1 + theta + grp. Return (b2, se2). NaN-safe on y."""
    m = np.isfinite(y)
    Xd = np.column_stack([np.ones(m.sum()), theta[m], grp[m]]); yv = y[m]
    if yv.sum() in (0, len(yv)) or grp[m].sum() in (0, m.sum()):
        return np.nan, np.nan
    beta = np.zeros(3)
    for _ in range(50):
        p = sig(Xd@beta); W = p*(1-p)+1e-9
        H = Xd.T@(W[:,None]*Xd) + ridge*np.eye(3)
        g = Xd.T@(yv-p) - ridge*beta
        step = np.linalg.solve(H, g); beta += step
        if np.max(np.abs(step)) < 1e-6: break
    cov = np.linalg.inv(Xd.T@(W[:,None]*Xd)+ridge*np.eye(3))
    return beta[2], np.sqrt(cov[2,2])


benches = list(dict.fromkeys(bench))
# leave-one-benchmark-out anchor ability
theta_anchor = {}
for B in benches:
    theta_anchor[B] = rasch_theta(X[:, bench != B])

rows = []
flagged = np.zeros(I, dtype=bool)
for B in benches:
    R = release[bench == B][0]
    exposed = (mrel >= R).astype(float)
    th = theta_anchor[B]
    cols = np.where(bench == B)[0]
    nflag = 0; b2s = []; ivw_num = 0.0; ivw_den = 0.0
    for j in cols:
        b2, se = lr_dif(X[:, j], th, exposed)
        if np.isfinite(b2) and se > 0 and se < 10:   # drop separated/degenerate items
            b2s.append(b2)
            w = 1.0/se**2; ivw_num += w*b2; ivw_den += w   # inverse-variance pooling
            if (b2/se > 1.96) and (b2 > 0.5):            # per-item flag (conservative)
                flagged[j] = True; nflag += 1
    pooled = ivw_num/ivw_den if ivw_den else float("nan")     # benchmark-level DIF effect
    pooled_se = (1.0/ivw_den)**0.5 if ivw_den else float("nan")
    pooled_z = pooled/pooled_se if ivw_den else float("nan")
    rows.append((B, R, int(exposed.sum()), int((1-exposed).sum()), len(cols),
                 nflag, 100*nflag/max(len(cols),1),
                 float(np.nanmean(b2s)) if b2s else float("nan"), pooled, pooled_z))

valid_rel = [x for x in mrel if x < "9999"]
rep = ["# Real anchor-DIF on HELM matrix", f"matrix: {M} models x {I} items",
       f"model release span: {min(valid_rel)} .. {max(valid_rel)}", "",
       "Per-item flags are conservative (clean group only 5-8 models). The benchmark-level",
       "pooled effect (inverse-variance weighted DIF, controls ability+item) is the powered test:",
       "pooled_b2 > 0 with |z|>1.96 => exposed models are systematically easier given ability.", "",
       f"{'bench':8s} {'rel':8s} {'#exp':>5s} {'#cln':>5s} {'items':>6s} {'flag':>5s} {'%flag':>6s} {'mean_b2':>8s} {'pooled':>7s} {'z':>7s}"]
for B,R,ne,nc,ni,nf,pf,mb,pl,pz in rows:
    rep.append(f"{B:8s} {R:8s} {ne:5d} {nc:5d} {ni:6d} {nf:5d} {pf:6.1f} {mb:8.3f} {pl:7.3f} {pz:7.1f}")

# ranking shift: full vs purified
theta_full = rasch_theta(X)
theta_pure = rasch_theta(X[:, ~flagged])
rank_full = (-theta_full).argsort().argsort()
rank_pure = (-theta_pure).argsort().argsort()
shift = rank_full - rank_pure
order = np.argsort(rank_full)
rep += ["", "# leaderboard shift after purifying flagged anchors (top movers)",
        f"{'model':40s} {'rel':8s} {'rk_full':>7s} {'rk_pure':>7s} {'shift':>6s}"]
for i in sorted(range(M), key=lambda i:-abs(shift[i]))[:15]:
    rep.append(f"{resp.index[i][:40]:40s} {mrel[i]:8s} {rank_full[i]+1:7d} {rank_pure[i]+1:7d} {shift[i]:+6d}")
rep += ["", f"items flagged total: {int(flagged.sum())}/{I} ({100*flagged.mean():.1f}%)",
        f"max |rank shift|: {int(np.abs(shift).max())}  | spearman(full,pure): "
        f"{np.corrcoef(rank_full,rank_pure)[0,1]:.4f}"]
txt = "\n".join(rep); open(os.path.join(OUT,f"results_helm_dif{TAG}.txt"),"w").write(txt+"\n")
print(txt)

# figures
fig, ax = plt.subplots(1, 2, figsize=(14,5.5))
bb=[r[0] for r in rows]; pl=[r[8] for r in rows]; pz=[r[9] for r in rows]; rl=[r[1] for r in rows]
ax[0].bar([f"{b}\n({r})" for b,r in zip(bb,rl)], pl, color="#C44E52")
for i,(p,z) in enumerate(zip(pl,pz)):
    ax[0].annotate(f"z={z:.0f}", (i,p), ha="center", va="bottom", fontsize=8)
ax[0].axhline(0, color="k", lw=.6)
mb_all=[r[7] for r in rows]
ax[0].set_ylabel("exposed-vs-clean DIF effect (logit)")
ax[0].set_title("Benchmark-level contamination DIF (exposed easier given ability)\n"
                "bar=IVW pooled (z shown); robust only if same sign as unweighted mean")
# overlay unweighted mean as markers to show sign agreement (robust) vs flip (not)
ax[0].plot(range(len(rows)), mb_all, "k_", ms=18, mew=2, label="unweighted mean b2")
ax[0].legend(fontsize=7, loc="best")
for i in range(M):
    ax[1].plot([0,1],[rank_full[i]+1,rank_pure[i]+1],"-",color="#bbb",lw=.6,zorder=1)
big=sorted(range(M),key=lambda i:-abs(shift[i]))[:8]
for i in big:
    ax[1].plot([0,1],[rank_full[i]+1,rank_pure[i]+1],"-o",lw=1.6,zorder=3)
    ax[1].annotate(resp.index[i][:18],(1.02,rank_pure[i]+1),fontsize=7,va="center")
ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["full anchor","purified"])
ax[1].invert_yaxis(); ax[1].set_ylabel("rank (1=best)"); ax[1].set_title("Leaderboard shift after purification")
fig.tight_layout(); fig.savefig(os.path.join(OUT,f"fig_helm_ranking_shift{TAG}.png"),dpi=130)
print(f"\nwrote results_helm_dif{TAG}.txt and fig_helm_ranking_shift{TAG}.png")
