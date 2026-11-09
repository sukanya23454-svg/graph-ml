import random

import numpy as np

from src.path_cover import greedy_path_cover, random_tracks


def _random_graph(n, seed):
    rs = random.Random(seed)
    nbrs = [set() for _ in range(n)]
    for i in range(1, n):
        j = rs.randrange(i)
        nbrs[i].add(j)
        nbrs[j].add(i)
    for _ in range(3):
        a, b = rs.randrange(n), rs.randrange(n)
        if a != b:
            nbrs[a].add(b)
            nbrs[b].add(a)
    return [sorted(s) for s in nbrs]


def test_cover_is_valid_partition_into_paths():
    for seed in range(5):
        n = 30
        nbrs = _random_graph(n, seed)
        tid, pos, k = greedy_path_cover(n, nbrs, random.Random(seed))
        assert len(set(tid.tolist())) == k
        for t in range(k):
            vs = sorted(np.where(tid == t)[0], key=lambda v: pos[v])
            assert [pos[v] for v in vs] == list(range(len(vs)))
            for a, b in zip(vs, vs[1:]):
                assert b in nbrs[a]


def test_swap_never_worse_than_noswap():
    n = 40
    nbrs = _random_graph(n, 1)
    _, _, k_swap = greedy_path_cover(n, nbrs, random.Random(7), swap=True)
    _, _, k_no = greedy_path_cover(n, nbrs, random.Random(7), swap=False)
    assert k_swap <= k_no


def test_path_graph_is_one_track():
    n = 10
    nbrs = [[j for j in (i - 1, i + 1) if 0 <= j < n] for i in range(n)]
    _, _, k = greedy_path_cover(n, nbrs, random.Random(0))
    assert k == 1


def test_random_tracks_preserves_track_sizes():
    n = 30
    nbrs = _random_graph(n, 2)
    tid, pos, _ = greedy_path_cover(n, nbrs, random.Random(0))
    tid2, pos2 = random_tracks(tid, pos, random.Random(1))
    assert sorted(zip(tid.tolist(), pos.tolist())) == sorted(zip(tid2.tolist(), pos2.tolist()))
