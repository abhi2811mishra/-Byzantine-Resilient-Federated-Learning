"""
demo.py  (Final Demonstration — Slide 18)

An interactive command-line demo for the BTech project presentation.

Usage:
  python demo.py                        # interactive mode (prompts for choices)
  python demo.py --preset quick         # run a quick preset comparison
  python demo.py --attack sign_flip --aggregator krum --byz_frac 0.3  # direct

The demo:
  1. Lets the user select dataset, clients, attack, aggregator, byz %
  2. Runs FL for the chosen number of rounds
  3. Displays accuracy/convergence graph
  4. Computes and prints Precision, Recall, F1, robustness, and timing
  5. Compares FedAvg vs all robust aggregation methods
  6. Explains which method works better and why
"""
import argparse, json, os, sys, time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fl import get_parser, run_experiment, SimpleCNN, load_mnist
from aggregators import AGGREGATORS
from attacks import ALL_ATTACKS


def interactive_select(prompt, options, default=None):
    """Let the user pick from a numbered list."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        marker = " (default)" if opt == default else ""
        print(f"  {i}. {opt}{marker}")
    while True:
        raw = input(f"  Enter choice [1-{len(options)}]: ").strip()
        if raw == "" and default:
            return default
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


def run_demo(args):
    """Run a single experiment and return detailed results."""
    print(f"\n{'='*60}")
    print(f"  Running: attack={args.attack}  aggregator={args.aggregator}")
    print(f"           byz_frac={args.byz_frac}  partition={args.partition}")
    print(f"           clients={args.clients}  rounds={args.rounds}")
    print(f"{'='*60}\n")

    hist, byz, clf, timing = run_experiment(args)

    return {
        "attack": args.attack,
        "aggregator": args.aggregator,
        "byz_frac": args.byz_frac,
        "partition": args.partition,
        "clients": args.clients,
        "accuracy": hist,
        "byzantine": byz,
        "metrics": clf,
        "timing": timing,
    }


def print_results(res):
    """Pretty-print a single result."""
    print(f"\n{'─'*50}")
    print(f"  Results: {res['attack']} + {res['aggregator']}")
    print(f"{'─'*50}")
    print(f"  Final accuracy : {res['accuracy'][-1]:.4f}")
    print(f"  Best accuracy  : {max(res['accuracy']):.4f}")
    m = res["metrics"]
    print(f"  Macro Precision: {m['macro_precision']:.4f}")
    print(f"  Macro Recall   : {m['macro_recall']:.4f}")
    print(f"  Macro F1-score : {m['macro_f1']:.4f}")
    t = res["timing"]
    print(f"  Total time     : {t['total_seconds']:.1f}s")
    print(f"  Avg round      : {t['avg_round_seconds']:.2f}s")
    print(f"  Avg aggregation: {t['avg_agg_seconds']*1000:.3f}ms")


def compare_and_explain(results, clean_acc=None):
    """Compare multiple aggregation methods and provide an explanation."""
    print(f"\n{'='*60}")
    print("  COMPARISON TABLE")
    print(f"{'='*60}")

    header = f"  {'Aggregator':<14} {'Final Acc':>10} {'Best Acc':>10} {'F1':>8} {'Time(s)':>8} {'Agg(ms)':>9}"
    print(header)
    print(f"  {'─'*14} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*9}")

    best_res = None
    best_acc = -1
    for r in results:
        final = r["accuracy"][-1]
        best = max(r["accuracy"])
        f1 = r["metrics"]["macro_f1"]
        total = r["timing"]["total_seconds"]
        agg_ms = r["timing"]["avg_agg_seconds"] * 1000
        print(f"  {r['aggregator']:<14} {final:>10.4f} {best:>10.4f} {f1:>8.4f} {total:>8.1f} {agg_ms:>9.3f}")
        if final > best_acc:
            best_acc = final
            best_res = r

    # Robustness metric (if clean baseline available)
    if clean_acc is not None:
        print(f"\n  Clean baseline accuracy: {clean_acc:.4f}")
        print(f"\n  {'Aggregator':<14} {'Acc Drop':>10} {'Recovery vs FedAvg':>20}")
        print(f"  {'─'*14} {'─'*10} {'─'*20}")
        fedavg_attacked = None
        for r in results:
            if r["aggregator"] == "fedavg":
                fedavg_attacked = r["accuracy"][-1]
                break
        for r in results:
            drop = clean_acc - r["accuracy"][-1]
            recovery = (r["accuracy"][-1] - fedavg_attacked) if fedavg_attacked is not None else 0
            print(f"  {r['aggregator']:<14} {drop:>+10.4f} {recovery:>+20.4f}")

    # Explanation
    print(f"\n{'─'*60}")
    print("  ANALYSIS")
    print(f"{'─'*60}")
    attack = results[0]["attack"]
    byz = results[0]["byz_frac"]

    if best_res:
        print(f"  ★ Best method: {best_res['aggregator']} "
              f"(accuracy={best_acc:.4f}, F1={best_res['metrics']['macro_f1']:.4f})")

    # Attack-specific explanations
    explanations = {
        "sign_flip": (
            "  Sign-flip is an UPDATE poisoning attack: malicious clients reverse\n"
            "  and scale their gradients. FedAvg averages these in, causing divergence.\n"
            "  Robust methods (Median, Trimmed Mean) filter out extreme values.\n"
            "  Krum selects the most 'central' update, ignoring outliers."
        ),
        "label_flip": (
            "  Label-flip is a DATA poisoning attack: malicious clients train on\n"
            "  incorrect labels (y → 9-y), producing subtly wrong gradients.\n"
            "  This is harder to detect because the magnitudes look normal.\n"
            "  Robust aggregators help but are less effective than against sign-flip."
        ),
        "gaussian": (
            "  Gaussian noise attack replaces malicious updates with random noise.\n"
            "  Median and Trimmed Mean handle this well by ignoring extreme values.\n"
            "  Krum can also reject these since noisy updates are far from honest ones."
        ),
        "ipm": (
            "  Inner Product Manipulation (IPM) is the most sophisticated attack\n"
            "  from the report. Attackers collude and send -ε × mean(honest updates).\n"
            "  This exploits the weakness of distance-based defences (Theorem 2).\n"
            "  Even Krum and Median can be defeated when ε is carefully chosen."
        ),
        "scaling": (
            "  Scaling attack amplifies honest updates by a factor, distorting\n"
            "  the aggregate. Median naturally resists this; Krum checks distances."
        ),
    }
    print(explanations.get(attack, "  No specific explanation available for this attack."))
    print()


def plot_comparison(results, filename="plots/demo_comparison.png"):
    """Plot accuracy curves for all methods."""
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    for r in results:
        rounds = range(1, len(r["accuracy"]) + 1)
        plt.plot(rounds, r["accuracy"], marker="o", ms=4, label=r["aggregator"])
    plt.xlabel("Communication Round")
    plt.ylabel("Global Test Accuracy")
    atk = results[0]["attack"]
    byz = results[0]["byz_frac"]
    part = results[0]["partition"]
    plt.title(f"Attack: {atk}, {int(byz*100)}% malicious, {part}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  → Saved comparison plot: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Byzantine FL Demo")
    parser.add_argument("--preset", choices=["quick", "full"],
                        help="Run a preset comparison instead of interactive mode")
    parser.add_argument("--attack", choices=ALL_ATTACKS, default=None)
    parser.add_argument("--aggregator", choices=sorted(AGGREGATORS), default=None)
    parser.add_argument("--byz_frac", type=float, default=None)
    parser.add_argument("--partition", choices=["iid", "noniid"], default=None)
    parser.add_argument("--clients", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--compare_all", action="store_true",
                        help="Compare all aggregators against the chosen attack")
    cli = parser.parse_args()

    # --- preset mode ---
    if cli.preset:
        attack = "sign_flip"
        byz_frac = 0.3
        partition = "iid"
        clients = 10
        rounds = 5 if cli.preset == "quick" else 15
        cli.compare_all = True
    elif cli.attack is None:
        # --- interactive mode ---
        print("╔══════════════════════════════════════════════════════════╗")
        print("║   Byzantine-Resilient Federated Learning — DEMO        ║")
        print("╚══════════════════════════════════════════════════════════╝")

        attack = interactive_select("Select attack type:", ALL_ATTACKS, default="sign_flip")
        partition = interactive_select("Select data distribution:", ["iid", "noniid"], default="iid")

        byz_options = ["0.1", "0.2", "0.3"]
        byz_frac = float(interactive_select("Select % of Byzantine clients:", byz_options, default="0.2"))

        client_options = ["10", "20", "50"]
        clients = int(interactive_select("Select number of clients:", client_options, default="10"))

        round_options = ["5", "10", "15", "20"]
        rounds = int(interactive_select("Select number of rounds:", round_options, default="10"))

        compare = interactive_select("Compare all aggregators?", ["yes", "no"], default="yes")
        cli.compare_all = (compare == "yes")

        if not cli.compare_all:
            agg = interactive_select("Select aggregation method:",
                                     sorted(AGGREGATORS.keys()), default="median")
        else:
            agg = None
    else:
        # --- direct CLI mode ---
        attack = cli.attack or "sign_flip"
        byz_frac = cli.byz_frac if cli.byz_frac is not None else 0.2
        partition = cli.partition or "iid"
        clients = cli.clients or 10
        rounds = cli.rounds or 10
        agg = cli.aggregator or "fedavg"

    # --- run clean baseline ---
    base_args = get_parser().parse_args([])
    base_args.attack = "none"
    base_args.aggregator = "fedavg"
    base_args.partition = partition
    base_args.clients = clients
    base_args.rounds = rounds
    base_args.byz_frac = 0.0

    print("\n▶ Running clean baseline (no attack, FedAvg)...")
    clean_res = run_demo(base_args)
    clean_acc = clean_res["accuracy"][-1]
    print(f"  Clean baseline accuracy: {clean_acc:.4f}")

    # --- run comparison ---
    results = []
    if cli.compare_all:
        for agg_name in sorted(AGGREGATORS.keys()):
            exp_args = get_parser().parse_args([])
            exp_args.attack = attack
            exp_args.aggregator = agg_name
            exp_args.byz_frac = byz_frac
            exp_args.partition = partition
            exp_args.clients = clients
            exp_args.rounds = rounds
            print(f"\n▶ Running {attack} + {agg_name}...")
            results.append(run_demo(exp_args))
    else:
        exp_args = get_parser().parse_args([])
        exp_args.attack = attack
        exp_args.aggregator = agg
        exp_args.byz_frac = byz_frac
        exp_args.partition = partition
        exp_args.clients = clients
        exp_args.rounds = rounds
        print(f"\n▶ Running {attack} + {agg}...")
        results.append(run_demo(exp_args))

    # --- display results ---
    for r in results:
        print_results(r)

    compare_and_explain(results, clean_acc=clean_acc)
    plot_comparison(results)

    # Save demo results
    os.makedirs("results", exist_ok=True)
    demo_out = {
        "clean_baseline": {"accuracy": clean_acc},
        "experiments": results,
    }
    with open("results/demo_results.json", "w") as f:
        json.dump(demo_out, f, indent=2, default=str)
    print(f"  → Saved: results/demo_results.json")


if __name__ == "__main__":
    main()

