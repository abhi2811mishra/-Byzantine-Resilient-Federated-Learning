"""
aggregators.py  (Month 3)

The SERVER's rule for combining client updates.

Every function takes `updates`: a numpy array of shape [n_clients, n_params]
(one flattened update per client) and returns ONE vector of shape [n_params].
Written in numpy on purpose so you can test them without PyTorch.

    fedavg        plain average (NOT robust - one attacker can ruin it)
    median        coordinate-wise median            (report, section 3)
    trimmed_mean  drop the f largest + f smallest per coordinate, average rest
    krum          pick the single update closest to its neighbours (report, section 3)
    multi_krum    average the m best-scoring updates by Krum score
"""
import numpy as np


def fedavg(updates, f=0):
    return updates.mean(axis=0)


def median(updates, f=0):
    return np.median(updates, axis=0)


def trimmed_mean(updates, f=0):
    n = updates.shape[0]
    if f == 0:
        return updates.mean(axis=0)
    if n - 2 * f < 1:
        raise ValueError(f"trimmed_mean needs n > 2f (n={n}, f={f})")
    s = np.sort(updates, axis=0)            # sort each coordinate independently
    return s[f:n - f].mean(axis=0)


def _krum_scores(updates, f):
    """For each update i: sum of squared distances to its n-f-2 nearest others."""
    n = updates.shape[0]
    k = n - f - 2
    if k < 1:
        raise ValueError(f"Krum needs n - 2f > 2 (n={n}, f={f})")
    sq = (updates ** 2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * updates @ updates.T   # squared distances
    d2 = np.maximum(d2, 0.0)
    scores = np.empty(n)
    for i in range(n):
        others = np.delete(d2[i], i)        # distances to everyone except itself
        scores[i] = np.sort(others)[:k].sum()
    return scores


def krum(updates, f=0):
    return updates[int(np.argmin(_krum_scores(updates, f)))]


def multi_krum(updates, f=0, m=None):
    n = updates.shape[0]
    m = m if m is not None else max(1, n - f)
    best = np.argsort(_krum_scores(updates, f))[:m]
    return updates[best].mean(axis=0)


AGGREGATORS = {
    "fedavg": fedavg,
    "median": median,
    "trimmed_mean": trimmed_mean,
    "krum": krum,
    "multi_krum": multi_krum,
}
