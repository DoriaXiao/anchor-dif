# -*- coding: utf-8 -*-
"""
Route-1 build: HELM Capabilities matrix (68 modern models, release 2023-12..2025-11).

Goal: an ABILITY-BALANCED contamination contrast. Omni-MATH (release 2024-10) splits these
modern frontier models into clean=15 (2023-12..2024-09) vs exposed=51 (2024-10+), both
high-ability -> good common support, unlike the old-benchmark contrasts in HELM Classic.
GPQA + MMLU-Pro are co-administered to the same 68 models and serve as the ability anchor.

CAVEAT: HELM release_date is the model's *release*, not training cutoff (usually months
earlier). For a tight window like Omni-MATH this mislabels some exposed models as having
seen it -> biases the exposure effect toward zero. Interpret accordingly.

Outputs in data_helm_cap/: helm_responses.csv, helm_items.csv, helm_models.csv (same schema
as 10_build_helm_matrix.py so 11_*.py can run on it).
"""
import os, re, json, urllib.request, urllib.parse, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "data_helm_cap"); os.makedirs(OUT, exist_ok=True)
GCS = "https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/"
REL = GCS + "releases/v1.15.0/"; RUN = GCS + "runs/"
SCEN = {
    "gpqa:":      ("GPQA",     "2023-11", "chain_of_thought_correctness", "model="),
    "mmlu_pro:":  ("MMLU-Pro", "2024-06", "chain_of_thought_correctness", "model="),
    "omni_math:": ("Omni-MATH","2024-10", "omni_math_accuracy",           "model="),
}
ITEM_ARGS = ("subset", "subject", "level", "task", "dataset")

def get(url, tries=4):
    for t in range(tries):
        try: return urllib.request.urlopen(url, timeout=120).read()
        except Exception:
            if t == tries-1: raise

r2s = json.loads(get(REL+"runs_to_run_suites.json"))
schema = json.loads(get(REL+"schema.json"))
relmap = {m["name"].replace("/","_"): m.get("release_date") for m in schema["models"]}

runs=[]
for pref,(bench,rel,metric,mh) in SCEN.items():
    for k,suite in r2s.items():
        if k.startswith(pref) and mh in k: runs.append((k,suite,bench,rel,metric))
print(f"{len(runs)} runs")

def variant(rn):
    args = rn.split(":",1)[1] if ":" in rn else ""
    kv={}
    for tok in args.split(","):
        if "=" in tok:
            k,v=tok.split("=",1)
            if k in ITEM_ARGS: kv[k]=v
    return ",".join(f"{k}={kv[k]}" for k in ITEM_ARGS if k in kv)

def model_of(rn):
    m=re.search(r"model=([^,]+)",rn); return m.group(1) if m else None

def fetch(rn,suite,bench,rel,metric):
    url=RUN+suite+"/"+urllib.parse.quote(rn)+"/per_instance_stats.json"
    d=json.loads(get(url)); var=variant(rn); mdl=model_of(rn); best={}
    for it in d:
        if it.get("train_trial_index",0)!=0: continue
        iid=it["instance_id"]
        for s in it["stats"]:
            nm=s["name"]
            if nm["name"]!=metric or nm.get("perturbation") is not None: continue
            best[iid]=float(s["mean"])
    return mdl,bench,rel,{f"{bench}|{var}|{iid}":(1.0 if v>=0.5 else 0.0) for iid,v in best.items()}

cells=collections.defaultdict(dict); item_meta={}; done=0
with ThreadPoolExecutor(max_workers=16) as ex:
    for f in as_completed([ex.submit(fetch,*r) for r in runs]):
        try: mdl,bench,rel,out=f.result()
        except Exception as e: print("ERR",e); continue
        cells[mdl].update(out)
        for iid in out: item_meta[iid]=(bench,rel)
        done+=1
        if done%50==0: print(f"  {done}/{len(runs)} models={len(cells)} items={len(item_meta)}")

models=sorted(cells); items=sorted(item_meta); ci={it:j for j,it in enumerate(items)}
X=np.full((len(models),len(items)),np.nan,np.float32)
for i,m in enumerate(models):
    for it,v in cells[m].items(): X[i,ci[it]]=v
pd.DataFrame(X,index=models,columns=items).to_csv(os.path.join(OUT,"helm_responses.csv"))
pd.DataFrame({"item_id":items,"benchmark":[item_meta[i][0] for i in items],
             "release":[item_meta[i][1] for i in items]}).to_csv(os.path.join(OUT,"helm_items.csv"),index=False)
pd.DataFrame({"model":models,"release_date":[relmap.get(m) for m in models],
             "n_items":[int(np.isfinite(X[i]).sum()) for i in range(len(models))],
             "accuracy":[float(np.nanmean(X[i])) for i in range(len(models))]}).to_csv(
    os.path.join(OUT,"helm_models.csv"),index=False)
print("done.", X.shape, "coverage %.1f%%"%(100*np.isfinite(X).mean()))
print("items/benchmark:", collections.Counter(item_meta[i][0] for i in items))
