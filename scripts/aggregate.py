"""
aggregate.py -- turn results/*.json into paper tables and the measured phase-diagram figure.

    python scripts/aggregate.py --task func --ref pcpe --base lap

Prints:
  1. overall test metric (mean +- std over seeds) for every variant
  2. metric per width / diameter / size bin for every variant
  3. PAIRED gap (ref - base, same seeds) per bin -- this is the Delta in Hypothesis 6.1,
     and the diameter / size panels are the controls the paper currently lacks.
Writes results_summary.csv and fig_gap_<ref>_vs_<base>.png.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


def ms(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--task", default="func")
    ap.add_argument("--ref", default="pcpe", help="variant whose gap we report")
    ap.add_argument("--base", default="lap", help="baseline variant")
    args = ap.parse_args()

    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.dir, "*.json")))]
    runs = [r for r in runs if r["task"] == args.task]
    if not runs:
        raise SystemExit("no matching results found")
    name = lambda r: r["pe"] + ("+bias" if r["bias"] else "")
    by = defaultdict(dict)                      # variant -> seed -> run
    for r in runs:
        by[name(r)][r["seed"]] = r

    metric_name = "AP (higher better)" if args.task == "func" else "MAE (lower better)"
    print(f"\n=== {args.task}: {metric_name} ===")
    print(f"{'variant':16s} {'n_seeds':>7s}   test (mean +- std)")
    rows = []
    for v, seeds in sorted(by.items()):
        m, s, k = ms([r["test"] for r in seeds.values()])
        print(f"{v:16s} {k:7d}   {m:.4f} +- {s:.4f}")
        rows.append(dict(variant=v, group="overall", bin="all", n_seeds=k, mean=m, std=s))

    groups = ["width", "diam", "size"]
    print("\n--- per-bin metric ---")
    for g in groups:
        print(f"\n[{g}]")
        bins = list(next(iter(next(iter(by.values())).values()))[g].keys())
        for b in bins:
            n_graphs = next(iter(next(iter(by.values())).values()))[g][b]["n"]
            line = f"  {b:12s} (n={n_graphs:4d}): "
            for v, seeds in sorted(by.items()):
                m, s, k = ms([r[g][b]["metric"] for r in seeds.values()])
                line += f"{v}={m:.4f}+-{s:.4f}  "
                rows.append(dict(variant=v, group=g, bin=b, n_seeds=k, mean=m, std=s))
            print(line)

    if args.ref not in by or args.base not in by:
        print(f"\n(need both '{args.ref}' and '{args.base}' for paired gaps; have {sorted(by)})")
    else:
        common = sorted(set(by[args.ref]) & set(by[args.base]))
        sign = 1.0 if args.task == "func" else -1.0      # positive gap always means ref is better
        print(f"\n=== paired gap, {args.ref} better than {args.base} (seeds {common}) ===")
        gaps = {}
        for g in groups:
            gaps[g] = {}
            bins = list(by[args.ref][common[0]][g].keys())
            for b in bins:
                d = [sign * (by[args.ref][s][g][b]["metric"] - by[args.base][s][g][b]["metric"])
                     for s in common
                     if by[args.ref][s][g][b]["metric"] is not None and by[args.base][s][g][b]["metric"] is not None]
                m, sd, k = ms(d)
                gaps[g][b] = (m, sd, k)
                print(f"  [{g}] {b:12s} gap = {m:+.4f} +- {sd:.4f}  (seeds: {k})")
                rows.append(dict(variant=f"gap:{args.ref}-{args.base}", group=g, bin=b, n_seeds=k, mean=m, std=sd))
        om, osd, ok = ms([sign * (by[args.ref][s]["test"] - by[args.base][s]["test"]) for s in common])
        print(f"  [overall] gap = {om:+.4f} +- {osd:.4f}")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
        titles = {"width": "by estimated width", "diam": "by diameter", "size": "by graph size"}
        for ax, g in zip(axes, groups):
            labels = list(gaps[g].keys())
            ms_ = [gaps[g][b][0] for b in labels]
            sd_ = [gaps[g][b][1] for b in labels]
            ax.errorbar(range(len(labels)), ms_, yerr=sd_, marker="o", capsize=3)
            ax.axhline(0, color="grey", ls="--", lw=0.8)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_title(titles[g], fontsize=10)
        axes[0].set_ylabel(f"gain of {args.ref} over {args.base}")
        fig.tight_layout()
        figname = f"fig_gap_{args.ref}_vs_{args.base}.png"
        fig.savefig(figname, dpi=200)
        print(f"\nSaved {figname}")

    with open("results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "group", "bin", "n_seeds", "mean", "std"])
        w.writeheader()
        w.writerows(rows)
    print("Saved results_summary.csv")


if __name__ == "__main__":
    main()
