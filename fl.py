"""
fl.py  (Months 1-4 combined)

The full federated learning simulator:
  data + partitions + model + client training + attack + aggregation + eval.

One round:
  1. every client trains a copy of the global model locally
     -> its UPDATE is  delta_i = local_weights - global_weights
  2. malicious clients tamper with their update (attacks.py)
  3. server aggregates the deltas (aggregators.py)
  4. global_weights += aggregated delta
  5. evaluate on the test set

Run one experiment:
  python fl.py --attack sign_flip --aggregator median --byz_frac 0.2
"""
import argparse, gzip, json, os, pickle, time, urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

from aggregators import AGGREGATORS
from attacks import apply_update_attack, DATA_ATTACKS, ALL_ATTACKS

# ------------- classification metrics (Precision, Recall, F1) ----------
def _compute_classification_metrics(model, loader, num_classes=10):
    """Compute per-class precision, recall, F1 and macro averages."""
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            for t, p in zip(y.numpy(), preds.numpy()):
                confusion[t, p] += 1

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (2 * precision[c] * recall[c] / (precision[c] + recall[c])
                 if (precision[c] + recall[c]) > 0 else 0.0)

    return {
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
    }

torch.set_num_threads(max(1, os.cpu_count() or 1))

# ----------------------------------------------------------------- data
MNIST_ROOT = "./mnist_data"
_FALLBACK_URL = ("https://raw.githubusercontent.com/mnielsen/"
                 "neural-networks-and-deep-learning/master/data/mnist.pkl.gz")


class _ArrayMNIST(Dataset):
    """Used only if the normal MNIST download is blocked on your network."""
    def __init__(self, images, labels, transform):
        self.images = (images.reshape(-1, 28, 28) * 255).astype(np.uint8)
        self.targets = torch.as_tensor(labels.astype(np.int64))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.targets[i])


def load_mnist():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    try:
        return (datasets.MNIST(MNIST_ROOT, train=True, download=True, transform=tf),
                datasets.MNIST(MNIST_ROOT, train=False, download=True, transform=tf))
    except Exception as e:
        print(f"[data] standard download failed ({e.__class__.__name__}); using fallback mirror")
        os.makedirs(MNIST_ROOT, exist_ok=True)
        path = os.path.join(MNIST_ROOT, "mnist_fallback.pkl.gz")
        if not os.path.exists(path):
            urllib.request.urlretrieve(_FALLBACK_URL, path)
        with gzip.open(path, "rb") as f:
            tr, va, te = pickle.load(f, encoding="latin1")
        return (_ArrayMNIST(np.concatenate([tr[0], va[0]]), np.concatenate([tr[1], va[1]]), tf),
                _ArrayMNIST(te[0], te[1], tf))


def partition(dataset, labels, n_clients, mode, samples_per_client, rng, shards_per_client=2):
    """Return a list of index arrays, one per client."""
    if mode == "iid":
        idx = rng.permutation(len(dataset))[: n_clients * samples_per_client]
        return np.array_split(idx, n_clients)
    # non-IID: sort by label, cut into shards, deal `shards_per_client` to each client
    n_shards = n_clients * shards_per_client
    shard_size = samples_per_client // shards_per_client
    out = []
    by_label = np.argsort(labels, kind="stable")
    big = len(by_label) // n_shards          # full-size shard in sorted order
    shards = []
    for s in range(n_shards):
        start = s * big
        shards.append(by_label[start:start + big][:shard_size])
    order = rng.permutation(n_shards)
    for c in range(n_clients):
        mine = order[c * shards_per_client:(c + 1) * shards_per_client]
        out.append(np.concatenate([shards[s] for s in mine]))
    return out


# ---------------------------------------------------------------- model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))


def get_flat(model):
    return parameters_to_vector(model.parameters()).detach().clone()


def set_flat(model, flat):
    # COPY the numbers in. (torch's vector_to_parameters would make the model
    # share memory with `flat`, so local training would silently overwrite it.)
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p.data))
        i += n


# --------------------------------------------------------------- client
def local_train(model, global_flat, dataset, args, flip_labels=False):
    """Train from the global weights; return the UPDATE (delta) as a numpy vector."""
    set_flat(model, global_flat)
    model.train()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(args.local_epochs):
        for x, y in loader:
            if flip_labels:
                y = 9 - y                       # label-flipping attack
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
    return (get_flat(model) - global_flat).numpy()


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


# ----------------------------------------------------------- experiment
def run_experiment(args):
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    train_set, test_set = load_mnist()
    labels = np.array(train_set.targets)
    n_test = args.test_samples or len(test_set)
    test_loader = DataLoader(Subset(test_set, list(range(n_test))), batch_size=512)

    parts = partition(train_set, labels, args.clients, args.partition, args.samples_per_client, rng)
    client_sets = [Subset(train_set, p.tolist()) for p in parts]

    q = int(round(args.byz_frac * args.clients))
    byz_idx = sorted(rng.choice(args.clients, q, replace=False).tolist()) if q > 0 else []
    if args.attack == "none":
        byz_idx = []
    print(f"attack={args.attack} aggregator={args.aggregator} partition={args.partition} "
          f"clients={args.clients} byzantine={byz_idx}")

    aggregate = AGGREGATORS[args.aggregator]
    model = SimpleCNN()
    global_flat = get_flat(model)
    history = []
    round_times = []         # wall-clock seconds per round
    agg_times = []           # aggregation time per round (seconds)
    t0 = time.time()
    for rnd in range(1, args.rounds + 1):
        t_rnd = time.time()
        updates = np.stack([
            local_train(model, global_flat, client_sets[i], args,
                        flip_labels=(args.attack in DATA_ATTACKS and i in byz_idx))
            for i in range(args.clients)
        ])
        updates = apply_update_attack(args.attack, updates, byz_idx,
                                      scale=args.attack_scale, eps=args.eps, rng=rng)
        t_agg = time.time()
        agg = aggregate(updates, f=q)
        agg_times.append(time.time() - t_agg)
        global_flat = global_flat + torch.from_numpy(agg).float()
        set_flat(model, global_flat)
        acc = evaluate(model, test_loader)
        history.append(acc)
        round_times.append(time.time() - t_rnd)
        print(f"  round {rnd:3d}/{args.rounds}  acc={acc:.4f}  ({time.time()-t0:.0f}s)")

    # Final classification metrics (Precision, Recall, F1)
    clf_metrics = _compute_classification_metrics(model, test_loader)
    total_time = time.time() - t0
    timing = {
        "total_seconds": round(total_time, 2),
        "avg_round_seconds": round(np.mean(round_times), 4),
        "avg_agg_seconds": round(np.mean(agg_times), 6),
        "round_times": [round(t, 4) for t in round_times],
        "agg_times": [round(t, 6) for t in agg_times],
    }
    return history, byz_idx, clf_metrics, timing


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", default="none", choices=ALL_ATTACKS)
    p.add_argument("--aggregator", default="fedavg", choices=sorted(AGGREGATORS))
    p.add_argument("--byz_frac", type=float, default=0.2)
    p.add_argument("--attack_scale", type=float, default=3.0)
    p.add_argument("--eps", type=float, default=0.1, help="strength of the ipm attack")
    p.add_argument("--partition", default="iid", choices=["iid", "noniid"])
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--samples_per_client", type=int, default=1000)
    p.add_argument("--test_samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="optional path to save result JSON")
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    hist, byz, clf_metrics, timing = run_experiment(args)
    print(f"final accuracy: {hist[-1]:.4f}")
    print(f"macro P={clf_metrics['macro_precision']:.4f}  "
          f"R={clf_metrics['macro_recall']:.4f}  F1={clf_metrics['macro_f1']:.4f}")
    print(f"total time: {timing['total_seconds']:.1f}s  "
          f"avg round: {timing['avg_round_seconds']:.2f}s  "
          f"avg agg: {timing['avg_agg_seconds']*1000:.3f}ms")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"args": vars(args), "byzantine": byz, "accuracy": hist,
                   "metrics": clf_metrics, "timing": timing}, open(args.out, "w"))
