"""
attacks.py  (Month 2)

Byzantine clients. Two kinds:

 DATA poisoning (happens during local training, handled in fl.py):
    label_flip   malicious clients train on flipped labels (y -> 9 - y)

 UPDATE poisoning (happens after training, handled here on the update vectors):
    sign_flip    send  -scale * (own honest update)
    gaussian     send random noise with a huge standard deviation
    scaling      send  scale * (own honest update)  (boosted)
    ipm          "inner product manipulation" from your report: all attackers
                 collude and send identical  -eps * mean(honest updates)
                 (omniscient: they see the honest updates)

`updates` is [n_clients, n_params]; `byz_idx` lists the malicious clients.
"""
import numpy as np

DATA_ATTACKS = {"label_flip"}
UPDATE_ATTACKS = {"none", "sign_flip", "gaussian", "scaling", "ipm"}
ALL_ATTACKS = sorted(DATA_ATTACKS | UPDATE_ATTACKS)


def apply_update_attack(name, updates, byz_idx, scale=3.0, eps=0.1, rng=None):
    out = updates.copy()
    if name in ("none", "label_flip") or len(byz_idx) == 0:
        return out
    rng = rng or np.random.RandomState(0)
    byz_idx = list(byz_idx)
    honest_idx = [i for i in range(len(updates)) if i not in set(byz_idx)]

    if name == "sign_flip":
        for i in byz_idx:
            out[i] = -scale * updates[i]
    elif name == "scaling":
        for i in byz_idx:
            out[i] = scale * updates[i]
    elif name == "gaussian":
        std = scale * (updates.std() + 1e-8) * 10
        for i in byz_idx:
            out[i] = rng.normal(0.0, std, size=updates.shape[1])
    elif name == "ipm":
        honest_mean = updates[honest_idx].mean(axis=0)
        for i in byz_idx:
            out[i] = -eps * honest_mean
    else:
        raise ValueError(f"unknown attack {name}")
    return out
