# -*- coding: utf-8 -*-
"""
Scale-up build: assemble a (models x items) binary matrix from HELM Classic per-instance
results, for benchmarks that give clean/exposed contrast (release dates vs model dates).

Source: public GCS bucket crfm-helm-public, classic release v0.4.0.
Scenarios: MMLU (rel 2020-09), GSM8K (2021-10), MATH (2021-03). ~67-69 models each,
model release dates 2019-2023 -> real clean (released-before-benchmark) reference groups.

Per-instance correctness metric differs by scenario:
  mmlu -> exact_match ; gsm -> exact_match_indicator ; math -> math_equiv  (all 0/1).
Item id = "<scenario>|<variant>|<instance_id>" where variant = run name minus the model=...
part, so the same item aligns across models. We keep the ORIGINAL (unperturbed) stat at
train_trial_index 0.

Outputs (in data_helm/):
  helm_responses.csv   models x items, 0/1, NaN where a model lacks an item
  helm_items.csv       item_id, benchmark, release
  helm_models.csv      model, release_date, n_items, accuracy
"""
import os, re, json, urllib.request, urllib.parse, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "data_helm")
os.makedirs(OUT, exist_ok=True)
GCS = "https://storage.googleapis.com/crfm-helm-public/classic/benchmark_output/"
REL = GCS + "releases/v0.4.0/"
RUN = GCS + "runs/"

SCEN = {  # scenario prefix -> (benchmark name, release, metric, extra filter substring)
    "mmlu:subject=": ("MMLU",   "2020-09", "exact_match",           "data_augmentation=canonical"),
    "gsm:":          ("GSM8K",  "2021-10", "exact_match_indicator", "model="),
    "math:":         ("MATH",   "2021-03", "math_equiv",            "use_chain_of_thought=False"),
}

# args that actually define a distinct ITEM (everything else -- model, method, prompt
# style, data_augmentation -- is not item-defining and must be excluded from the item key).
ITEM_ARGS = ("subject", "level", "task", "dataset")

def get(url, tries=4):
    for t in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=120).read()
        except Exception as e:
            if t == tries - 1:
                raise
    return None

print("loading run->suite map ...")
r2s = json.loads(get(REL + "runs_to_run_suites.json"))
schema = json.loads(get(REL + "schema.json"))
# HELM run names use "_" where schema model names use "/" (e.g. openai/davinci -> openai_davinci)
relmap = {m["name"].replace("/", "_"): m.get("release_date") for m in schema["models"]}

# enumerate target runs
runs = []   # (run_name, suite, benchmark, release, metric)
for pref, (bench, rel, metric, mh) in SCEN.items():
    for k, suite in r2s.items():
        if k.startswith(pref) and mh in k:
            runs.append((k, suite, bench, rel, metric))
print(f"{len(runs)} runs to fetch across {len(SCEN)} scenarios")

def variant(run_name):
    # keep only item-defining args (subject/level/task/dataset); drop scenario name, model,
    # prompt style, data_augmentation, etc. This makes item ids align across models AND
    # across non-item-defining prompt conditions.
    args = run_name.split(":", 1)[1] if ":" in run_name else ""
    kv = {}
    for tok in args.split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            if k in ITEM_ARGS:
                kv[k] = v
    return ",".join(f"{k}={kv[k]}" for k in ITEM_ARGS if k in kv)

def model_of(run_name):
    m = re.search(r"model=([^,]+)", run_name)
    return m.group(1) if m else None

def fetch(run_name, suite, bench, rel, metric):
    url = RUN + suite + "/" + urllib.parse.quote(run_name) + "/per_instance_stats.json"
    d = json.loads(get(url))
    var = variant(run_name); mdl = model_of(run_name)
    best = {}  # instance_id -> value (original perturbation, trial 0)
    for it in d:
        if it.get("train_trial_index", 0) != 0:
            continue
        iid = it["instance_id"]
        for s in it["stats"]:
            nm = s["name"]
            if nm["name"] != metric:
                continue
            if nm.get("perturbation") is not None:
                continue
            best[iid] = float(s["mean"])
    out = {}
    for iid, v in best.items():
        out[f"{bench}|{var}|{iid}"] = (1.0 if v >= 0.5 else 0.0)
    return mdl, bench, rel, out

cells = collections.defaultdict(dict)   # model -> {item_id: 0/1}
item_meta = {}                          # item_id -> (benchmark, release)
done = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = [ex.submit(fetch, *r) for r in runs]
    for f in as_completed(futs):
        try:
            mdl, bench, rel, out = f.result()
        except Exception as e:
            print("  ERR:", e); continue
        cells[mdl].update(out)
        for iid in out:
            item_meta[iid] = (bench, rel)
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(runs)} runs, models={len(cells)}, items={len(item_meta)}")

models = sorted(cells)
items = sorted(item_meta)
print(f"\nbuilding matrix {len(models)} models x {len(items)} items ...")
col_idx = {it: j for j, it in enumerate(items)}
X = np.full((len(models), len(items)), np.nan, dtype=np.float32)
for i, m in enumerate(models):
    for it, v in cells[m].items():
        X[i, col_idx[it]] = v

resp = pd.DataFrame(X, index=models, columns=items)
resp.to_csv(os.path.join(OUT, "helm_responses.csv"))
pd.DataFrame({"item_id": items,
             "benchmark": [item_meta[i][0] for i in items],
             "release":  [item_meta[i][1] for i in items]}).to_csv(
    os.path.join(OUT, "helm_items.csv"), index=False)
pd.DataFrame({"model": models,
             "release_date": [relmap.get(m) for m in models],
             "n_items": [int(np.isfinite(X[i]).sum()) for i in range(len(models))],
             "accuracy": [float(np.nanmean(X[i])) for i in range(len(models))]}).to_csv(
    os.path.join(OUT, "helm_models.csv"), index=False)

print("done. wrote helm_responses.csv, helm_items.csv, helm_models.csv to data_helm/")
print("matrix coverage: %.1f%% cells observed" % (100*np.isfinite(X).mean()))
print("items per benchmark:", collections.Counter(item_meta[i][0] for i in items))
