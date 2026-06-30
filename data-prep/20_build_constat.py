# -*- coding: utf-8 -*-
"""
Pivot foundation: build the ConStat normal-vs-rephrase response matrices.

Source: ConStat (NeurIPS 2024) public model outputs compressed_output.zip. Each model has
per-instance CSVs per (benchmark, variant) with a 0/1 correctness column + doc_id:
  gsm8k -> exact_match ; mmlu -> acc ; arc/hellaswag -> acc_norm.
normal and rephrase share doc_id (rephrase = semantically-equivalent paraphrase of the same
item) -> aligning on doc_id gives a within-skill control that isolates MEMORIZATION from
capability. That is the clean identification the HELM/release_date route lacked.

Two model groups:
  - REAL (47): released models (Qwen1.5, Llama-2/3, Mistral/Mixtral, Phi, Yi, Gemma, ...).
  - CONTROLLED (74): Llama-2-7b{,-chat} finetuned with KNOWN contamination dose per benchmark
    (default=test-set leak, rephrase=rephrased-test leak, repeat_1, lr_1e-*, training=train-
    split-only [negative control]). -> ground-truth labels to calibrate anchor-DIF.

Outputs to data_constat/:
  {bench}_{variant}.csv   models x doc_id (0/1, NaN if absent), variant in {normal,rephrase,synthetic}
  models_meta.csv         model, group(real/controlled), base, target_bench, condition
"""
import os, re, io, csv, zipfile, collections
import numpy as np, pandas as pd

ZIP = "/private/tmp/claude-503/-Users-doriax/c3aafee4-1338-48ee-a6d4-55035588e377/scratchpad/constat_output.zip"
OUT = os.path.join(os.path.dirname(__file__), "data_constat"); os.makedirs(OUT, exist_ok=True)
BENCHES = ["arc", "gsm8k", "hellaswag", "mmlu"]
VARIANTS = ["normal", "rephrase", "synthetic"]
METRIC = {"gsm8k": "exact_match", "mmlu": "acc", "arc": "acc_norm", "hellaswag": "acc_norm"}

z = zipfile.ZipFile(ZIP)
csvs = [n for n in z.namelist() if n.endswith(".csv")]

def model_of(n):  # compressed_output/<org>/<model>/<variant>/file.csv
    p = n.split("/"); return p[1] + "/" + p[2]

def variant_of(n):
    return n.split("/")[3]

def read_doc_acc(name, metric):
    with z.open(name) as f:
        rdr = csv.DictReader(io.TextIOWrapper(f, "utf-8"))
        out = {}
        for row in rdr:
            if metric in row and row[metric] != "" and row.get("doc_id", "") != "":
                try:
                    out[int(float(row["doc_id"]))] = 1.0 if float(row[metric]) >= 0.5 else 0.0
                except ValueError:
                    pass
    return out

# index files by (model, bench, variant)
idx = collections.defaultdict(dict)   # (bench,variant) -> {model: filename}
for n in csvs:
    v = variant_of(n)
    for b in BENCHES:
        if v == f"{b}_normal" or v == f"{b}_rephrase" or v == f"{b}_synthetic":
            kind = v.split("_", 1)[1]
            idx[(b, kind)][model_of(n)] = n

for (b, kind), mp in sorted(idx.items()):
    metric = METRIC[b]
    cols_data = {m: read_doc_acc(fn, metric) for m, fn in mp.items()}
    all_docs = sorted(set().union(*[set(d) for d in cols_data.values()])) if cols_data else []
    models = sorted(cols_data)
    M = np.full((len(models), len(all_docs)), np.nan, np.float32)
    ci = {d: j for j, d in enumerate(all_docs)}
    for i, m in enumerate(models):
        for d, v in cols_data[m].items():
            M[i, ci[d]] = v
    pd.DataFrame(M, index=models, columns=all_docs).to_csv(os.path.join(OUT, f"{b}_{kind}.csv"))
    print(f"{b}_{kind:9s}: {len(models)} models x {len(all_docs)} items, coverage {100*np.isfinite(M).mean():.0f}%")

# model metadata
rows = []
for m in sorted(set().union(*[set(mp) for mp in idx.values()])):
    if m.startswith("JasperDekoninck/contamination-models-"):
        tail = m.split("contamination-models-")[1]
        mb = re.match(r"(arc|gsm8k|hellaswag|mmlu)-(.*)", tail)
        bench, cond = (mb.group(1), mb.group(2)) if mb else ("?", tail)
        base = "llama2-7b-chat" if "Llama-2-7b-chat" in cond else "llama2-7b"
        cond = cond.replace("meta-llama-Llama-2-7b-chat-hf-", "")
        rows.append((m, "controlled", base, bench, cond))
    else:
        rows.append((m, "real", "", "", ""))
pd.DataFrame(rows, columns=["model", "group", "base", "target_bench", "condition"]).to_csv(
    os.path.join(OUT, "models_meta.csv"), index=False)
print(f"\nmodels_meta.csv: {sum(r[1]=='real' for r in rows)} real, {sum(r[1]=='controlled' for r in rows)} controlled")
