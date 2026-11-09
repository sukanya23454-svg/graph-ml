import hashlib
import inspect
import os
import pickle
import random
import time

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data

from src.path_cover import greedy_path_cover, random_tracks
from src.positional_encodings import K_LAP, K_RW

COVER_SEED = 12345


# =============================================================================
# Per-graph precomputation (all PE variants + width/diameter), cached on disk
# =============================================================================
def precompute_graph(data, idx):
    n = data.num_nodes
    ei = data.edge_index.numpy()
    A = np.zeros((n, n), dtype=np.float64)
    A[ei[0], ei[1]] = 1.0
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    nbrs = [np.nonzero(A[v])[0].tolist() for v in range(n)]
    deg = A.sum(1)

    # LapPE: eigenvectors 1..k of the symmetric normalised Laplacian
    dinv = np.where(deg > 0, np.power(np.maximum(deg, 1e-12), -0.5), 0.0)
    L = np.eye(n) - dinv[:, None] * A * dinv[None, :]
    _, evecs = np.linalg.eigh(L)
    vec = evecs[:, 1:K_LAP + 1]
    lap = np.zeros((n, K_LAP), dtype=np.float32)
    lap[:, :vec.shape[1]] = vec

    # RWSE: diagonal of P^t, t = 1..K_RW
    P = A / np.maximum(deg, 1.0)[:, None]
    M = P.copy()
    rw = np.zeros((n, K_RW), dtype=np.float32)
    for t in range(K_RW):
        rw[:, t] = np.diag(M)
        M = M @ P

    # diameter (largest finite distance)
    dist = shortest_path(csr_matrix(A), unweighted=True, directed=False)
    diam = int(np.max(dist[np.isfinite(dist)])) if n > 1 else 0

    # path covers (deterministic per graph)
    seed = COVER_SEED * 1_000_003 + idx
    tid_s, pos_s, k_s = greedy_path_cover(n, nbrs, random.Random(seed), swap=True)
    tid_n, pos_n, k_n = greedy_path_cover(n, nbrs, random.Random(seed), swap=False)
    tid_r, pos_r = random_tracks(tid_s, pos_s, random.Random(seed + 7))
    return dict(lap=lap, rw=rw, diam=diam, n=n,
                swap=(tid_s, pos_s, k_s), noswap=(tid_n, pos_n, k_n),
                rand=(tid_r, pos_r, k_s))


# =============================================================================
# Data
# =============================================================================
def fake_splits(task):
    rs = np.random.RandomState(0)

    def g():
        n = int(rs.randint(12, 60))
        edges = [(i, int(rs.randint(0, i))) for i in range(1, n)]
        for _ in range(int(rs.randint(0, 4))):
            a, b = int(rs.randint(0, n)), int(rs.randint(0, n))
            if a != b:
                edges.append((a, b))
        ei = torch.tensor(edges + [(b, a) for a, b in edges]).t().contiguous()
        x = torch.randint(0, 5, (n, 9))
        ea = torch.randint(0, 3, (ei.size(1), 3))
        y = (torch.rand(1, 10) > 0.8).float() if task == "func" else torch.randn(1, 11)
        return Data(x=x, edge_index=ei, edge_attr=ea, y=y)

    return [[g() for _ in range(m)] for m in (160, 40, 40)]


def load_splits(task, root, fake):
    if fake:
        return fake_splits(task)
    from torch_geometric.datasets import LRGBDataset
    name = "Peptides-func" if task == "func" else "Peptides-struct"
    out = []
    for split in ("train", "val", "test"):
        ds = LRGBDataset(root, name, split=split)
        out.append([ds[i] for i in range(len(ds))])
    return out


def cache_version():
    src = "".join(inspect.getsource(f) for f in (greedy_path_cover, random_tracks, precompute_graph))
    src += f"|{K_LAP}|{K_RW}|{COVER_SEED}"
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def get_caches(splits, task, root, fake):
    path = os.path.join(root, f"pe_cache_{task}{'_fake' if fake else ''}.pkl")
    version = cache_version()
    if os.path.exists(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and obj.get("version") == version:
            return obj["caches"]
        print("PE cache is missing a version or is outdated; rebuilding.", flush=True)
    print("Precomputing PEs / path covers (one-off, cached)...", flush=True)
    t0 = time.time()
    caches = []
    for si, split in enumerate(splits):
        caches.append([precompute_graph(d, si * 10_000_000 + i) for i, d in enumerate(split)])
        print(f"  split {si}: {len(split)} graphs, {time.time() - t0:.0f}s", flush=True)
    os.makedirs(root, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"version": version, "caches": caches}, f)
    return caches


def attach(split, cache, pe):
    out = []
    for d, c in zip(split, cache):
        d = d.clone()
        d.width = torch.tensor([c["swap"][2]])
        d.diam = torch.tensor([c["diam"]])
        if pe == "lap":
            d.pe = torch.from_numpy(c["lap"])
        elif pe == "rwse":
            d.pe = torch.from_numpy(c["rw"])
        elif pe in ("pcpe", "pcpe_noswap", "pcpe_random"):
            key = {"pcpe": "swap", "pcpe_noswap": "noswap", "pcpe_random": "rand"}[pe]
            d.pcid = torch.from_numpy(c[key][0])
            d.pcpos = torch.from_numpy(c[key][1])
        out.append(d)
    return out
