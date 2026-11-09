"""
plot_results.py  (Month 4-5)

Turns results/*.json into report figures in plots/:
  curves_<attack>_<partition>.png   accuracy vs round, one line per aggregator
  final_acc_<partition>.png         heatmap-style table: final accuracy, attack x aggregator
"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("plots", exist_ok=True)
runs = [json.load(open(p)) for p in glob.glob("results/*.json")]
runs = [r for r in runs if "attack" in r]
if not runs:
    raise SystemExit("no results found - run run_matrix.py first")

parts = sorted({r["partition"] for r in runs})
attacks = sorted({r["attack"] for r in runs if r["attack"] != "none"})
aggs = sorted({r["aggregator"] for r in runs})
fracs = sorted({r["byz_frac"] for r in runs if r["attack"] != "none"})
base = {r["partition"]: r["accuracy"] for r in runs if r["attack"] == "none"}

for part in parts:
    for atk in attacks:
        fr = fracs[-1]                                   # show the strongest attacker fraction
        plt.figure(figsize=(6, 4))
        for agg in aggs:
            for r in runs:
                if (r["attack"], r["aggregator"], r["partition"], r["byz_frac"]) == (atk, agg, part, fr):
                    plt.plot(range(1, len(r["accuracy"]) + 1), r["accuracy"], marker="o", ms=3, label=agg)
        if part in base:
            plt.plot(range(1, len(base[part]) + 1), base[part], "k--", label="no attack (FedAvg)")
        plt.xlabel("round"); plt.ylabel("test accuracy"); plt.grid(True, alpha=.3)
        plt.title(f"{atk}, {int(fr*100)}% malicious, {part}")
        plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(f"plots/curves_{atk}_{part}.png", dpi=140); plt.close()

    for fr in fracs:
        M = np.full((len(attacks), len(aggs)), np.nan)
        for r in runs:
            if r["partition"] == part and r["byz_frac"] == fr and r["attack"] in attacks:
                M[attacks.index(r["attack"]), aggs.index(r["aggregator"])] = r["accuracy"][-1]
        plt.figure(figsize=(1.3 * len(aggs) + 2, 0.7 * len(attacks) + 1.8))
        plt.imshow(M, vmin=0, vmax=1, cmap="RdYlGn")
        plt.xticks(range(len(aggs)), aggs, rotation=30, ha="right")
        plt.yticks(range(len(attacks)), attacks)
        for i in range(len(attacks)):
            for j in range(len(aggs)):
                if not np.isnan(M[i, j]):
                    plt.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(label="final accuracy")
        plt.title(f"{part}, {int(fr*100)}% malicious")
        plt.tight_layout(); plt.savefig(f"plots/final_acc_{part}_{int(fr*100)}pct.png", dpi=140); plt.close()
print("saved figures in plots/")
