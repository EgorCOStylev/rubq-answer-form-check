import numpy as np
from metrics import group_sc


def sc_uncertainty(samples, kind, fn):
    return round(1.0 - group_sc(samples, kind, fn), 9)


def _avg_ranks(x):
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auroc(u, err):
    u = np.asarray(u, float)
    err = np.asarray(err, bool)
    n1, n0 = int(err.sum()), int((~err).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((_avg_ranks(u)[err].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _aurc_sorted(err_sorted):
    n = len(err_sorted)
    return float((np.cumsum(err_sorted) / np.arange(1, n + 1)).mean())


def aurc_naive(u, err):
    u, err = np.asarray(u, float), np.asarray(err, float)
    return _aurc_sorted(err[np.argsort(u, kind="stable")])


def aurc_bounds(u, err):
    u, err = np.asarray(u, float), np.asarray(err, float)
    best = np.lexsort((err, u))
    worst = np.lexsort((-err, u))
    return _aurc_sorted(err[best]), _aurc_sorted(err[worst])


def aurc_tie_aware(u, err):
    u, err = np.asarray(u, float), np.asarray(err, float)
    n = len(u)
    cum_err, seen, total = 0.0, 0, 0.0
    for v in np.unique(u):
        m = u == v
        size, e = int(m.sum()), float(err[m].sum())
        j = np.arange(1, size + 1)
        total += float(((cum_err + j * e / size) / (seen + j)).sum())
        cum_err += e
        seen += size
    return total / n


def aurc_ideal(err):
    return _aurc_sorted(np.sort(np.asarray(err, float)))
