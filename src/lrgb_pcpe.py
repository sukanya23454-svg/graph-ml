"""
lrgb_pcpe.py -- entry point to (re)run the LRGB experiments.

Trains a GPS-style model (GINE local MPNN + full softmax attention) on
Peptides-func (AP) or Peptides-struct (MAE) with one positional encoding at a
time, then evaluates the test set overall AND stratified by estimated width
(w_H_hat), diameter, and graph size.

PE variants (--pe):
  none          no positional encoding
  lap           LapPE, k=8 Laplacian eigenvectors (random sign flip in training)
  rwse          RWSE, k=16 random-walk return probabilities
  pcpe          PC-PE feature form: learned track-id embedding + sinusoidal position
  pcpe_noswap   same, but the greedy cover WITHOUT the local-swap merging pass
  pcpe_random   same, but tracks are randomly permuted (keeps track sizes, breaks
                adjacency) -- the control for "does path structure matter?"
Add --bias (with a pcpe* variant) to ALSO add the attention-bias form
b(u,v) = g[pos(u)-pos(v)] for same-track pairs. Needs more GPU memory.

Run from the repository root.

Quick check that everything runs (no download, seconds on CPU):
    python -m src.lrgb_pcpe --fake --epochs 2 --pe pcpe --bias

Real run (downloads Peptides through PyG on first use):
    python -m src.lrgb_pcpe --task func --pe pcpe --seed 0 --epochs 100
"""
import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.datasets import attach, get_caches, load_splits
from src.evaluation import predict
from src.metrics import DIAM_BINS, WIDTH_BINS, metric, stratify
from src.models import Net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="func", choices=["func", "struct"])
    ap.add_argument("--pe", default="pcpe",
                    choices=["none", "lap", "rwse", "pcpe", "pcpe_noswap", "pcpe_random"])
    ap.add_argument("--bias", action="store_true", help="also add the attention-bias form (pcpe* only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--attn_dropout", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default=None,
                    help="result folder (default: experiments/results/peptides_<task>)")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--max_minutes", type=float, default=None,
                    help="stop training after this many minutes; best-val checkpoint is still used")
    ap.add_argument("--force", action="store_true", help="rerun even if the result file exists")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join("experiments", "results", f"peptides_{args.task}")
    if args.bias and not args.pe.startswith("pcpe"):
        raise SystemExit("--bias needs a pcpe* variant")
    tag = f"{args.task}_{args.pe}{'_bias' if args.bias else ''}_seed{args.seed}"
    if os.path.exists(os.path.join(args.out, tag + ".json")) and not args.force:
        print(f"SKIP {tag}: result already exists (use --force to rerun)")
        return

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = load_splits(args.task, args.root, args.fake)
    caches = get_caches(splits, args.task, args.root, args.fake)
    all_x = torch.cat([d.x for s in splits for d in s])
    all_e = torch.cat([d.edge_attr for s in splits for d in s])
    node_sizes = (all_x.max(0).values + 1).tolist()
    edge_sizes = (all_e.max(0).values + 1).tolist()
    out_dim = splits[0][0].y.shape[1]

    data = [attach(s, c, args.pe) for s, c in zip(splits, caches)]
    tr = DataLoader(data[0], batch_size=args.bs, shuffle=True)
    va = DataLoader(data[1], batch_size=args.bs)
    te = DataLoader(data[2], batch_size=args.bs)

    model = Net(node_sizes, edge_sizes, out_dim, args.pe, args.bias, args.dim, args.layers,
                args.heads, args.dropout, args.attn_dropout).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"task={args.task} pe={args.pe} bias={args.bias} seed={args.seed} "
          f"params={n_params:,} device={dev}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def lr_lambda(ep):
        if ep < args.warmup:
            return (ep + 1) / args.warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - args.warmup) / max(1, args.epochs - args.warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    higher_better = args.task == "func"
    best = (-1e9 if higher_better else 1e9)
    best_ep, best_val, best_state = -1, None, None

    t_start, capped, epochs_run = time.time(), False, 0
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for b in tr:
            b = b.to(dev)
            opt.zero_grad()
            out = model(b)
            loss = (F.binary_cross_entropy_with_logits(out, b.y) if args.task == "func"
                    else F.l1_loss(out, b.y))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * b.num_graphs
        sched.step()

        Pv, Yv = predict(model, va, dev)
        v = metric(args.task, Pv, Yv)
        better = v > best if higher_better else v < best
        if better:
            best, best_ep, best_val = v, ep, v
            best_state = copy.deepcopy(model.state_dict())
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep {ep:3d} loss {tot / len(data[0]):.4f} val {v:.4f} "
                  f"({time.time() - t0:.1f}s/epoch)", flush=True)
        epochs_run = ep + 1
        if args.max_minutes is not None and (time.time() - t_start) / 60 > args.max_minutes:
            capped = True
            print(f"TIME CAP reached after {epochs_run} of {args.epochs} epochs", flush=True)
            break

    # test set is evaluated exactly once, on the checkpoint selected by validation
    model.load_state_dict(best_state)
    best_P, Y_test = predict(model, te, dev)
    best_test = metric(args.task, best_P, Y_test)

    widths = np.array([d.width.item() for d in data[2]])
    diams = np.array([d.diam.item() for d in data[2]])
    sizes = np.array([d.num_nodes for d in data[2]])
    q = np.quantile(sizes, [1 / 3, 2 / 3])
    size_bins = [(0, float(q[0])), (float(q[0]), float(q[1])), (float(q[1]), 10 ** 9)]

    result = {
        "task": args.task, "pe": args.pe, "bias": args.bias, "seed": args.seed,
        "epochs": args.epochs, "epochs_run": epochs_run, "time_capped": capped, "best_epoch": best_ep, "val": best_val, "test": best_test,
        "params": n_params,
        "peak_gpu_gb": (torch.cuda.max_memory_allocated() / 2 ** 30) if dev.type == "cuda" else 0.0,
        "width": stratify(args.task, best_P, Y_test, widths, WIDTH_BINS),
        "diam": stratify(args.task, best_P, Y_test, diams, DIAM_BINS),
        "size": stratify(args.task, best_P, Y_test, sizes, size_bins),
        "config": vars(args),
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, tag + ".json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"DONE {tag}: test={best_test:.4f} (best val epoch {best_ep}), "
          f"peak GPU memory {result['peak_gpu_gb']:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
