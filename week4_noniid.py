"""
week4_noniid.py
Goal (Month 1, Week 4): see how FedAvg behaves when clients do NOT have
the same kind of data.

  IID      : every client gets a random slice -> sees all 10 digits
  non-IID  : data is sorted by digit, cut into shards, and each client
             gets only 2 shards -> sees only ~2 digits (the "pathological"
             split from the original FedAvg paper, also used in your report)

We run the SAME FedAvg loop on both and plot them together.
This matters later: Byzantine defences like Median and Krum look for
"outlier" updates, but in non-IID data honest clients already look
different from each other, which makes attackers much harder to spot.

Run: python week4_noniid.py
"""
import copy
import os
import gzip
import pickle
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
NUM_CLIENTS = 10
NUM_ROUNDS = int(os.environ.get("FL_ROUNDS", 10))
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LR = 0.01
SHARDS_PER_CLIENT = 2     # non-IID: how many label-shards each client gets
SEED = 0

# ------------------------------------------------------------------
# 1. DATA (same loader as weeks 2-3)
# ------------------------------------------------------------------
MNIST_ROOT = "./mnist_data"
_FALLBACK_URL = ("https://raw.githubusercontent.com/mnielsen/"
                 "neural-networks-and-deep-learning/master/data/mnist.pkl.gz")


class _ArrayMNIST(Dataset):
    """Only used if the normal MNIST download is blocked on your network."""
    def __init__(self, images, labels, transform):
        self.images = (images.reshape(-1, 28, 28) * 255).astype(np.uint8)
        self.targets = torch.as_tensor(labels.astype(np.int64))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.targets[idx])


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    try:
        train = datasets.MNIST(MNIST_ROOT, train=True, download=True, transform=transform)
        test = datasets.MNIST(MNIST_ROOT, train=False, download=True, transform=transform)
        return train, test
    except Exception as e:
        print(f"[data] standard download failed ({e.__class__.__name__}); using fallback mirror")
        os.makedirs(MNIST_ROOT, exist_ok=True)
        path = os.path.join(MNIST_ROOT, "mnist_fallback.pkl.gz")
        if not os.path.exists(path):
            urllib.request.urlretrieve(_FALLBACK_URL, path)
        with gzip.open(path, "rb") as f:
            tr, va, te = pickle.load(f, encoding="latin1")
        tx = np.concatenate([tr[0], va[0]])
        ty = np.concatenate([tr[1], va[1]])
        return _ArrayMNIST(tx, ty, transform), _ArrayMNIST(te[0], te[1], transform)


train_set, test_set = load_mnist()
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
train_labels = np.array(train_set.targets)


# ------------------------------------------------------------------
# 2. TWO WAYS TO SPLIT DATA ACROSS CLIENTS
# ------------------------------------------------------------------
def split_iid(dataset, num_clients, rng):
    idx = rng.permutation(len(dataset))
    shards = np.array_split(idx, num_clients)
    return [Subset(dataset, s.tolist()) for s in shards]


def split_noniid(dataset, labels, num_clients, shards_per_client, rng):
    """
    1. sort all sample indices by digit label  (all 0s, then all 1s, ...)
    2. cut into num_clients * shards_per_client equal shards
    3. deal the shards out randomly, shards_per_client to each client
    """
    sorted_idx = np.argsort(labels, kind="stable")
    num_shards = num_clients * shards_per_client
    shard_size = len(dataset) // num_shards
    shards = [sorted_idx[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]

    order = rng.permutation(num_shards)
    subsets = []
    for c in range(num_clients):
        mine = order[c * shards_per_client:(c + 1) * shards_per_client]
        idx = np.concatenate([shards[s] for s in mine])
        subsets.append(Subset(dataset, idx.tolist()))
    return subsets


def show_label_counts(name, client_datasets, labels):
    print(f"\n{name} split - digits each client holds (digit:count):")
    for c, ds in enumerate(client_datasets):
        digits, counts = np.unique(labels[ds.indices], return_counts=True)
        desc = ", ".join(f"{d}:{n}" for d, n in zip(digits, counts))
        print(f"  client {c}: {desc}")


# ------------------------------------------------------------------
# 3. MODEL (same CNN)
# ------------------------------------------------------------------
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
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ------------------------------------------------------------------
# 4. CLIENT + SERVER (same as week 3)
# ------------------------------------------------------------------
def local_train(global_model, dataset):
    model = copy.deepcopy(global_model)
    model.train()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    return model.state_dict()


def fedavg(client_states):
    avg = copy.deepcopy(client_states[0])
    for key in avg.keys():
        avg[key] = torch.stack([s[key].float() for s in client_states]).mean(dim=0)
    return avg


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total


def run_fedavg(client_datasets, label):
    torch.manual_seed(SEED)           # same starting weights for a fair comparison
    global_model = SimpleCNN()
    history = []
    for rnd in range(1, NUM_ROUNDS + 1):
        states = [local_train(global_model, ds) for ds in client_datasets]
        global_model.load_state_dict(fedavg(states))
        acc = evaluate(global_model, test_loader)
        history.append(acc)
        print(f"[{label}] round {rnd}/{NUM_ROUNDS} - test accuracy: {acc:.4f}")
    return history


# ------------------------------------------------------------------
# 5. RUN BOTH EXPERIMENTS
# ------------------------------------------------------------------
rng = np.random.RandomState(SEED)
iid_clients = split_iid(train_set, NUM_CLIENTS, rng)
noniid_clients = split_noniid(train_set, train_labels, NUM_CLIENTS, SHARDS_PER_CLIENT, rng)

show_label_counts("IID", iid_clients, train_labels)
show_label_counts("non-IID", noniid_clients, train_labels)

print("\n=== Training FedAvg on IID data ===")
iid_history = run_fedavg(iid_clients, "IID")
print("\n=== Training FedAvg on non-IID data ===")
noniid_history = run_fedavg(noniid_clients, "non-IID")

# ------------------------------------------------------------------
# 6. PLOT
# ------------------------------------------------------------------
rounds = range(1, NUM_ROUNDS + 1)
plt.figure()
plt.plot(rounds, iid_history, marker="o", label="IID")
plt.plot(rounds, noniid_history, marker="s", label="non-IID (2 digits per client)")
plt.xlabel("Communication round")
plt.ylabel("Global test accuracy")
plt.title(f"FedAvg: IID vs non-IID ({NUM_CLIENTS} clients)")
plt.legend()
plt.grid(True)
plt.savefig("fedavg_iid_vs_noniid.png", dpi=150, bbox_inches="tight")
print(f"\nFinal accuracy - IID: {iid_history[-1]:.4f} | non-IID: {noniid_history[-1]:.4f}")
print("Saved: fedavg_iid_vs_noniid.png")