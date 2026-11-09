"""
seed_summary.py - mean +/- std of FINAL accuracy over seeds, for non-IID, 30% malicious.

Reads
  results/<attack>__<agg>__f0.3__noniid__s0.json     (seed 0, from run_matrix.py)
  results/seed_<attack>_<agg>_s<seed>.json            (extra seeds, from fl.py --out)
Run:  python seed_summary.py
"""
import glob, json, os
import numpy as np

ATTACKS = ["ipm", "label_flip", "sign_flip"]
AGGS = ["fedavg", "median", "trimmed_mean", "krum", "multi_krum"]

def final(path):
    return json.load(open(path))["accuracy"][-1]

print(f"{'attack':<11}" + "".join(f"{a:>20}" for a in AGGS))
for atk in ATTACKS:
    row = f"{atk:<11}"
    for agg in AGGS:
        vals = []
        p0 = f"results/{atk}__{agg}__f0.3__noniid__s0.json"
        if os.path.exists(p0):
            vals.append(final(p0))
        for p in sorted(glob.glob(f"results/seed_{atk}_{agg}_s*.json")):
            vals.append(final(p))
        cell = "-" if not vals else "%.3f+/-%.3f (n=%d)" % (np.mean(vals), np.std(vals), len(vals))
        row += f"{cell:>20}"
    print(row)