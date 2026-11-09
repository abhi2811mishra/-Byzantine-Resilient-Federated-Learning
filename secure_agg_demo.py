"""
secure_agg_demo.py  (Month 5, optional extension - numpy only)

A SIMPLIFIED SIMULATION of the privacy-robustness paradox from your report.
It is NOT real cryptography and NOT the BREA protocol (BREA uses Lagrange
coded computing, verifiable secret sharing, etc.). It shows the idea:

 Part 1  additive masking: the server sees only random-looking vectors, yet the
         masks cancel in the SUM, so it still learns the exact aggregate.
 Part 2  the paradox: robust rules (Krum/median) need to look at individual
         updates, but masked updates are indistinguishable from noise.
 Part 3  the BREA idea, simulated: the server learns ONLY the pairwise distances
         (via an oracle standing in for the cryptographic protocol), excludes
         outliers, then securely sums just the accepted updates.
"""
import numpy as np
from aggregators import _krum_scores

rng = np.random.RandomState(0)
n, d, f = 8, 6, 2
P = 2 ** 31 - 1          # a prime: we work in a finite field, like the report says
SCALE = 10_000           # stochastic-quantisation stand-in: float -> integer

true = rng.normal(0.5, 0.1, size=(n, d))       # honest-looking updates
true[:f] = -5.0 * true[:f]                      # 2 attackers send large opposite updates
q = np.round(true * SCALE).astype(np.int64) % P  # quantise into the field

# ---- Part 1: pairwise additive masks ------------------------------------
def make_masks(rng):
    masks = np.zeros((n, d), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            secret = rng.randint(0, P, size=d, dtype=np.int64)   # secret shared by pair (i, j)
            masks[i] = (masks[i] + secret) % P                   # i adds it
            masks[j] = (masks[j] - secret) % P                   # j subtracts it
    return masks

masks = make_masks(rng)
masked = (q + masks) % P

print("Part 1 - what the server sees for client 0 (looks random):", masked[0][:3])
total = masked.sum(axis=0) % P
true_total = q.sum(axis=0) % P
print("  masks cancel in the sum:", np.array_equal(total, true_total))

def decode(v):                                   # field element -> signed float
    v = np.where(v > P // 2, v - P, v)
    return v / SCALE

print("  decoded sum:", np.round(decode(total), 3))
print("  plain sum  :", np.round(true.sum(axis=0), 3))

# ---- Part 2: the paradox -----------------------------------------------
# Run Krum on the masked vectors with many different random masks and count how
# often it "selects" an attacker (clients 0 and 1). Pure chance would be 2/8 = 25%.
picked_attacker = 0
trials = 300
for _ in range(trials):
    m = (q + make_masks(rng)) % P
    chosen = int(np.argmin(_krum_scores(m.astype(float), f)))
    picked_attacker += chosen in (0, 1)
print(f"\nPart 2 - Krum on MASKED vectors picks an attacker {100*picked_attacker/trials:.0f}% "
      f"of the time (chance level is 25%): masked updates carry no usable signal")

# ---- Part 3: verifiable distance check (simulated oracle) ---------------
print("\nPart 3 - Krum scores from the distance oracle (reveals only distances):")
scores = _krum_scores(true, f)
print("  ", np.round(scores, 2))
keep = np.argsort(scores)[: n - f]
print("  accepted clients:", sorted(keep.tolist()), "(attackers 0,1 excluded:", not ({0, 1} & set(keep.tolist())), ")")
accepted_sum = masked[keep].sum(axis=0) % P
# All masks sum to zero over ALL clients, so the accepted clients' masks sum to the
# NEGATIVE of the excluded clients' masks. A real protocol recovers that correction
# from secret shares; here we recompute it directly to show the idea.
mask_excl = masks[[i for i in range(n) if i not in keep]].sum(axis=0) % P
secure_total = (accepted_sum + mask_excl) % P
print("  secure aggregate of accepted updates:", np.round(decode(secure_total), 3))
print("  plain sum of honest updates         :", np.round(true[keep].sum(axis=0), 3))
