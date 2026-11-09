"""
run_matrix.py  (Month 4)

Runs the experiment grid  attack x aggregator x byzantine-fraction x partition
and saves one JSON per run in results/ plus a combined results/summary.csv.
Already-finished runs are skipped, so you can stop (Ctrl+C) and restart safely.

  python run_matrix.py --quick     # small grid, a few rounds: sanity check
  python run_matrix.py             # the full grid for your report (takes hours on CPU)
"""
import argparse, csv, glob, json, os

from fl import get_parser, run_experiment

QUICK = dict(attacks=["sign_flip", "ipm"], aggs=["fedavg", "median", "krum"],
             fracs=[0.2], parts=["iid"], rounds=3, seeds=[0])
FULL = dict(attacks=["label_flip", "sign_flip", "gaussian", "scaling", "ipm"],
            aggs=["fedavg", "median", "trimmed_mean", "krum", "multi_krum"],
            fracs=[0.1, 0.2, 0.3], parts=["iid", "noniid"], rounds=15, seeds=[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--results_dir", default="results")
    cli = ap.parse_args()
    grid = QUICK if cli.quick else FULL
    os.makedirs(cli.results_dir, exist_ok=True)

    jobs = [("none", "fedavg", 0.0, part, seed) for part in grid["parts"] for seed in grid["seeds"]]  # clean baselines
    for part in grid["parts"]:
        for atk in grid["attacks"]:
            for agg in grid["aggs"]:
                for fr in grid["fracs"]:
                    for seed in grid["seeds"]:
                        jobs.append((atk, agg, fr, part, seed))

    for n, (atk, agg, fr, part, seed) in enumerate(jobs, 1):
        name = f"{atk}__{agg}__f{fr}__{part}__s{seed}"
        path = os.path.join(cli.results_dir, name + ".json")
        if os.path.exists(path) and len(json.load(open(path))["accuracy"]) == grid["rounds"]:
            print(f"[{n}/{len(jobs)}] skip {name}")
            continue
        print(f"[{n}/{len(jobs)}] {name}")
        args = get_parser().parse_args([])
        args.attack, args.aggregator, args.byz_frac = atk, agg, fr
        args.partition, args.seed, args.rounds = part, seed, grid["rounds"]
        hist, byz, clf_metrics, timing = run_experiment(args)
        json.dump({"attack": atk, "aggregator": agg, "byz_frac": fr, "partition": part,
                   "seed": seed, "byzantine": byz, "accuracy": hist,
                   "metrics": clf_metrics, "timing": timing}, open(path, "w"))

    rows = []
    for p in sorted(glob.glob(os.path.join(cli.results_dir, "*.json"))):
        r = json.load(open(p))
        if "attack" not in r:
            continue
        rows.append([r["attack"], r["aggregator"], r["byz_frac"], r["partition"], r["seed"],
                     round(r["accuracy"][-1], 4), round(max(r["accuracy"]), 4)])
    with open(os.path.join(cli.results_dir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attack", "aggregator", "byz_frac", "partition", "seed", "final_acc", "best_acc"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {cli.results_dir}/summary.csv")


if __name__ == "__main__":
    main()
