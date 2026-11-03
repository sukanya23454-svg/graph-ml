import numpy as np
from sklearn.metrics import average_precision_score

WIDTH_BINS = [(1, 5), (5, 20), (20, 10 ** 9)]
DIAM_BINS = [(1, 10), (10, 30), (30, 10 ** 9)]


def metric(task, P, Y):
    if task == "func":
        aps = [average_precision_score(Y[:, c], P[:, c]) for c in range(Y.shape[1]) if Y[:, c].sum() > 0]
        return float(np.mean(aps)) if aps else float("nan")
    return float(np.abs(P - Y).mean())


def stratify(task, P, Y, vals, bins):
    res = {}
    for lo, hi in bins:
        idx = np.where((vals >= lo) & (vals < hi))[0]
        key = f"[{lo},{'inf' if hi >= 10 ** 9 else hi})"
        res[key] = {"n": int(len(idx)), "metric": metric(task, P[idx], Y[idx]) if len(idx) > 0 else None}
    return res
