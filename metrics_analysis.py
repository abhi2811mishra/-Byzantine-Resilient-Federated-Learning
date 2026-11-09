"""
metrics_analysis.py  (Month 4-5)

Computes the three derived metrics from the project plan (Slide 13):

  1. Robustness  = accuracy drop under attack  (clean − attacked)
  2. Recovery    = improvement from robust agg vs attacked FedAvg
  3. Efficiency  = aggregation time comparison across methods

Also prints Precision / Recall / F1 when available in the result JSONs.

Reads: results/*.json  (produced by run_matrix.py)
Writes:
  results/metrics_report.txt    human-readable summary
  results/robustness.csv        robustness numbers
  results/recovery.csv          recovery numbers
  results/efficiency.csv        timing numbers
  plots/robustness_heatmap.png  robustness visualisation
  plots/recovery_heatmap.png    recovery visualisation

Run:  python metrics_analysis.py
"""
import glob, json, os
import numpy as np

# ---- load results -------------------------------------------------------
runs = {}
for p in sorted(glob.glob("results/*.json")):
    r = json.load(open(p))
    if "attack" not in r:
        continue
    key = (r["attack"], r["aggregator"], r["byz_frac"], r["partition"])
    runs[key] = r

if not runs:
    raise SystemExit("no results found - run run_matrix.py first")

attacks = sorted({k[0] for k in runs if k[0] != "none"})
aggs = sorted({k[1] for k in runs})
fracs = sorted({k[2] for k in runs if k[0] != "none"})
parts = sorted({k[3] for k in runs})

# ---- clean baselines ----------------------------------------------------
clean = {}
for part in parts:
    key = ("none", "fedavg", 0.0, part)
    if key in runs:
        clean[part] = runs[key]["accuracy"][-1]
    else:
        clean[part] = None

os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

lines = []
lines.append("=" * 78)
lines.append("  BYZANTINE-RESILIENT FL — FULL METRICS REPORT")
lines.append("=" * 78)

# ---- 1. ROBUSTNESS: accuracy drop under attack -------------------------
lines.append("\n1. ROBUSTNESS — Accuracy Drop Under Attack")
lines.append("   (clean_accuracy − attacked_accuracy; higher = more damage)")
lines.append("-" * 78)

rob_rows = []
for part in parts:
    if clean[part] is None:
        continue
    lines.append(f"\n  Partition: {part}  |  Clean baseline accuracy: {clean[part]:.4f}")
    header = f"  {'attack':<12} {'byz%':>5}"
    for agg in aggs:
        header += f" {agg:>13}"
    lines.append(header)

    for atk in attacks:
        for fr in fracs:
            row_str = f"  {atk:<12} {int(fr*100):>4}%"
            for agg in aggs:
                key = (atk, agg, fr, part)
                if key in runs:
                    attacked_acc = runs[key]["accuracy"][-1]
                    drop = clean[part] - attacked_acc
                    row_str += f" {drop:>+13.4f}"
                    rob_rows.append([atk, agg, fr, part, round(clean[part], 4),
                                    round(attacked_acc, 4), round(drop, 4)])
                else:
                    row_str += f" {'N/A':>13}"
            lines.append(row_str)

import csv
with open("results/robustness.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["attack", "aggregator", "byz_frac", "partition",
                "clean_acc", "attacked_acc", "accuracy_drop"])
    w.writerows(rob_rows)
lines.append(f"\n  → Saved: results/robustness.csv ({len(rob_rows)} rows)")

# ---- 2. RECOVERY: robust agg improvement over attacked FedAvg -----------
lines.append("\n\n2. RECOVERY — Improvement Over Attacked FedAvg")
lines.append("   (robust_agg_accuracy − fedavg_attacked_accuracy; higher = better defence)")
lines.append("-" * 78)

rec_rows = []
for part in parts:
    lines.append(f"\n  Partition: {part}")
    header = f"  {'attack':<12} {'byz%':>5}"
    for agg in aggs:
        if agg == "fedavg":
            header += f" {'fedavg(ref)':>13}"
        else:
            header += f" {agg:>13}"
    lines.append(header)

    for atk in attacks:
        for fr in fracs:
            fedavg_key = (atk, "fedavg", fr, part)
            fedavg_acc = runs[fedavg_key]["accuracy"][-1] if fedavg_key in runs else None
            row_str = f"  {atk:<12} {int(fr*100):>4}%"
            for agg in aggs:
                key = (atk, agg, fr, part)
                if agg == "fedavg":
                    row_str += f" {fedavg_acc:>13.4f}" if fedavg_acc else f" {'N/A':>13}"
                elif key in runs and fedavg_acc is not None:
                    robust_acc = runs[key]["accuracy"][-1]
                    recovery = robust_acc - fedavg_acc
                    row_str += f" {recovery:>+13.4f}"
                    rec_rows.append([atk, agg, fr, part, round(fedavg_acc, 4),
                                    round(robust_acc, 4), round(recovery, 4)])
                else:
                    row_str += f" {'N/A':>13}"
            lines.append(row_str)

with open("results/recovery.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["attack", "aggregator", "byz_frac", "partition",
                "fedavg_acc", "robust_acc", "recovery"])
    w.writerows(rec_rows)
lines.append(f"\n  → Saved: results/recovery.csv ({len(rec_rows)} rows)")

# ---- 3. EFFICIENCY: timing comparison -----------------------------------
lines.append("\n\n3. EFFICIENCY — Training & Aggregation Time")
lines.append("-" * 78)

eff_rows = []
has_timing = False
for part in parts:
    lines.append(f"\n  Partition: {part}")
    header = f"  {'attack':<12} {'byz%':>5} {'aggregator':<14} {'total_s':>9} {'avg_rnd_s':>10} {'avg_agg_ms':>11}"
    lines.append(header)
    for atk in ["none"] + attacks:
        for fr in ([0.0] if atk == "none" else fracs):
            for agg in (["fedavg"] if atk == "none" else aggs):
                key = (atk, agg, fr, part)
                if key not in runs:
                    continue
                r = runs[key]
                t = r.get("timing")
                if t:
                    has_timing = True
                    lines.append(f"  {atk:<12} {int(fr*100):>4}% {agg:<14} "
                                 f"{t['total_seconds']:>9.2f} "
                                 f"{t['avg_round_seconds']:>10.4f} "
                                 f"{t['avg_agg_seconds']*1000:>11.3f}")
                    eff_rows.append([atk, agg, fr, part,
                                    t["total_seconds"], t["avg_round_seconds"],
                                    round(t["avg_agg_seconds"]*1000, 3)])

if not has_timing:
    lines.append("  (no timing data found — re-run run_matrix.py with updated fl.py)")

if eff_rows:
    with open("results/efficiency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attack", "aggregator", "byz_frac", "partition",
                    "total_seconds", "avg_round_seconds", "avg_agg_ms"])
        w.writerows(eff_rows)
    lines.append(f"\n  → Saved: results/efficiency.csv ({len(eff_rows)} rows)")

# ---- 4. PRECISION / RECALL / F1 (if available) -------------------------
lines.append("\n\n4. CLASSIFICATION METRICS — Precision, Recall, F1 (macro-averaged)")
lines.append("-" * 78)

has_clf = False
for part in parts:
    lines.append(f"\n  Partition: {part}")
    header = f"  {'attack':<12} {'byz%':>5} {'aggregator':<14} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    lines.append(header)
    for atk in ["none"] + attacks:
        for fr in ([0.0] if atk == "none" else fracs):
            for agg in (["fedavg"] if atk == "none" else aggs):
                key = (atk, agg, fr, part)
                if key not in runs:
                    continue
                m = runs[key].get("metrics")
                if m:
                    has_clf = True
                    lines.append(f"  {atk:<12} {int(fr*100):>4}% {agg:<14} "
                                 f"{m['macro_precision']:>10.4f} "
                                 f"{m['macro_recall']:>10.4f} "
                                 f"{m['macro_f1']:>10.4f}")

if not has_clf:
    lines.append("  (no classification metrics found — re-run run_matrix.py with updated fl.py)")

# ---- write report -------------------------------------------------------
report = "\n".join(lines)
print(report)
with open("results/metrics_report.txt", "w") as f:
    f.write(report)
print(f"\n→ Full report saved: results/metrics_report.txt")

# ---- 5. PLOTS -----------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for part in parts:
        # Robustness heatmap (worst case: highest byz_frac)
        fr = fracs[-1]
        M = np.full((len(attacks), len(aggs)), np.nan)
        for i, atk in enumerate(attacks):
            for j, agg in enumerate(aggs):
                key = (atk, agg, fr, part)
                if key in runs and clean[part] is not None:
                    M[i, j] = clean[part] - runs[key]["accuracy"][-1]

        plt.figure(figsize=(1.3 * len(aggs) + 2, 0.7 * len(attacks) + 1.8))
        plt.imshow(M, vmin=0, vmax=0.8, cmap="YlOrRd")
        plt.xticks(range(len(aggs)), aggs, rotation=30, ha="right")
        plt.yticks(range(len(attacks)), attacks)
        for i in range(len(attacks)):
            for j in range(len(aggs)):
                if not np.isnan(M[i, j]):
                    plt.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7)
        plt.colorbar(label="accuracy drop")
        plt.title(f"Robustness: accuracy drop ({part}, {int(fr*100)}% malicious)")
        plt.tight_layout()
        plt.savefig(f"plots/robustness_heatmap_{part}_{int(fr*100)}pct.png", dpi=140)
        plt.close()

        # Recovery heatmap
        M2 = np.full((len(attacks), len(aggs)), np.nan)
        for i, atk in enumerate(attacks):
            fedavg_key = (atk, "fedavg", fr, part)
            fa = runs[fedavg_key]["accuracy"][-1] if fedavg_key in runs else None
            for j, agg in enumerate(aggs):
                key = (atk, agg, fr, part)
                if key in runs and fa is not None:
                    M2[i, j] = runs[key]["accuracy"][-1] - fa

        plt.figure(figsize=(1.3 * len(aggs) + 2, 0.7 * len(attacks) + 1.8))
        vmax = max(0.1, np.nanmax(np.abs(M2)))
        plt.imshow(M2, vmin=-vmax, vmax=vmax, cmap="RdYlGn")
        plt.xticks(range(len(aggs)), aggs, rotation=30, ha="right")
        plt.yticks(range(len(attacks)), attacks)
        for i in range(len(attacks)):
            for j in range(len(aggs)):
                if not np.isnan(M2[i, j]):
                    plt.text(j, i, f"{M2[i, j]:+.3f}", ha="center", va="center", fontsize=7)
        plt.colorbar(label="recovery (vs attacked FedAvg)")
        plt.title(f"Recovery: robust agg vs FedAvg ({part}, {int(fr*100)}% malicious)")
        plt.tight_layout()
        plt.savefig(f"plots/recovery_heatmap_{part}_{int(fr*100)}pct.png", dpi=140)
        plt.close()

    print("→ Saved: plots/robustness_heatmap_*.png, plots/recovery_heatmap_*.png")
except Exception as e:
    print(f"(plot generation skipped: {e})")

