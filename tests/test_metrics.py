import numpy as np

from src.metrics import metric, stratify


def test_struct_metric_is_mae():
    P = np.array([[1.0], [3.0]])
    Y = np.array([[0.0], [1.0]])
    assert metric("struct", P, Y) == 1.5


def test_func_metric_perfect_ranking():
    Y = np.array([[1], [0], [1], [0]])
    P = np.array([[0.9], [0.1], [0.8], [0.2]])
    assert metric("func", P, Y) == 1.0


def test_stratify_bins():
    P = np.zeros((4, 1))
    Y = np.array([[0.0], [1.0], [2.0], [3.0]])
    vals = np.array([1, 4, 6, 25])
    res = stratify("struct", P, Y, vals, [(1, 5), (5, 20), (20, 10 ** 9)])
    assert [res[k]["n"] for k in res] == [2, 1, 1]
