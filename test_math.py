"""
test_math.py - checks the aggregators and attacks against the exact toy
examples in your report. Needs only numpy:  python test_math.py
"""
import numpy as np
from aggregators import fedavg, median, krum, trimmed_mean, multi_krum, _krum_scores
from attacks import apply_update_attack

col = lambda xs: np.array(xs, dtype=float).reshape(-1, 1)

# --- 1. plain average is destroyed by ONE attacker -----------------------
honest = [0.2, 0.4, 0.9]
print("fedavg honest only:", fedavg(col(honest))[0])
print("fedavg + one attacker (100):", fedavg(col(honest + [100]))[0])
assert fedavg(col(honest + [100]))[0] > 20

# --- 2. Theorem 1 toy example: median is flipped -------------------------
V, U = [-0.1, 0.1, 0.3], [-4, -2]
m = median(col(V + U))[0]
print("Theorem 1: true mean =", np.mean(V), " median(V+U) =", m)
assert abs(m - (-0.1)) < 1e-9 and np.sign(m) != np.sign(np.mean(V))

# --- 3. Theorem 2 toy example: Krum picks the attacker -------------------
V = [0, 0.02, 0.14, 0.26, 0.38, 0.5]
U = [-0.1, -0.1, -0.1]
X = col(V + U)
sc = _krum_scores(X, f=3)
print("Theorem 2: KR(byzantine -0.1) =", round(sc[-1], 4), " KR(honest 0) =", round(sc[0], 4))
assert abs(sc[-1] - 0.0244) < 1e-6 and abs(sc[0] - 0.0304) < 1e-6
assert krum(X, f=3)[0] == -0.1
print("Krum selects:", krum(X, f=3)[0], "(a malicious vector)")

# --- 4. trimmed mean resists the big-value attacker ----------------------
tm = trimmed_mean(col(honest + [100]), f=1)[0]
print("trimmed_mean with attacker:", tm)
assert tm < 1.0

# --- 5. Krum rejects a LARGE outlier (matches report: Krum fine at big eps)
X = col([0.1, 0.12, 0.09, 0.11, 0.1, 0.13, 50, 50])
assert abs(krum(X, f=2)[0]) < 1

# --- 6. attack functions -------------------------------------------------
U = np.ones((5, 4))
assert (apply_update_attack("sign_flip", U, [0], scale=2)[0] == -2).all()
ipm = apply_update_attack("ipm", U, [3, 4], eps=0.1)
assert np.allclose(ipm[3], -0.1) and np.allclose(ipm[4], ipm[3]) and np.allclose(ipm[0], 1)
print("\nALL TESTS PASSED")
