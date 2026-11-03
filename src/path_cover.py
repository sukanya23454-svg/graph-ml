import numpy as np


# =============================================================================
# Greedy path cover (Appendix B.1)
# =============================================================================
def greedy_path_cover(n, nbrs, rng, swap=True):
    """Returns (track_id, position, k). Tracks are numbered by size, largest first.
    Greedy pass: each vertex makes at most one join. Local swap: repeat full
    passes that join any two paths with adjacent endpoints, until none remain."""
    path_of = list(range(n))
    paths = {v: [v] for v in range(n)}

    def is_end(v):
        p = paths[path_of[v]]
        return p[0] == v or p[-1] == v

    def join(v, u):
        kv, ku = path_of[v], path_of[u]
        pv, pu = paths[kv], paths[ku]
        if pv[-1] != v:
            pv = pv[::-1]
        if pu[0] != u:
            pu = pu[::-1]
        del paths[ku]
        paths[kv] = pv + pu
        for w in pu:
            path_of[w] = kv

    order = list(range(n))
    rng.shuffle(order)
    for v in order:
        if not is_end(v):
            continue
        cand = list(nbrs[v])
        rng.shuffle(cand)
        for u in cand:
            if path_of[u] != path_of[v] and is_end(u):
                join(v, u)
                break

    if swap:
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if not is_end(v):
                    continue
                for u in nbrs[v]:
                    if path_of[u] != path_of[v] and is_end(u):
                        join(v, u)
                        changed = True
                        break

    tracks = sorted(paths.values(), key=lambda p: (-len(p), p[0]))
    tid = np.zeros(n, dtype=np.int64)
    pos = np.zeros(n, dtype=np.int64)
    for i, p in enumerate(tracks):
        for j, v in enumerate(p):
            tid[v] = i
            pos[v] = j
    return tid, pos, len(tracks)


def random_tracks(tid, pos, rng):
    """Control: permute which node gets which (track, position) slot."""
    n = len(tid)
    perm = list(range(n))
    rng.shuffle(perm)
    tid2, pos2 = np.zeros_like(tid), np.zeros_like(pos)
    for v_new, v_old in enumerate(perm):
        tid2[v_new] = tid[v_old]
        pos2[v_new] = pos[v_old]
    return tid2, pos2
