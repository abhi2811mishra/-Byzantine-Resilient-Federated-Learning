"""
week3_fedavg.py
Goal (Task 4 from the plan): build the basic FedAvg workflow from scratch.

One communication round =
  1. SERVER broadcasts the current global model to the clients
  2. each CLIENT trains that model on ITS OWN local data only
  3. each client sends its updated weights back
  4. SERVER averages the weights  (this average is "FedAvg")
  5. SERVER evaluates the new global model on the test set

Raw data never leaves a client: only weights travel.

Run: python week3_fedavg.py
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
# Settings you can play with
# ------------------------------------------------------------------
NUM_CLIENTS = 10      # how many simulated devices
NUM_ROUNDS = int(os.environ.get("FL_ROUNDS", 10))   # communication rounds
LOCAL_EPOCHS = 1      # epochs each client trains per round
BATCH_SIZE = 32
LR = 0.01
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------------
# 1. DATA (same loader as week 2)
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


# ------------------------------------------------------------------
# 2. SPLIT THE DATA ACROSS CLIENTS (IID = every client gets a random,
#    equal-sized slice, so each one sees all 10 digits)
# ------------------------------------------------------------------
def split_iid(dataset, num_clients):
    idx = np.random.permutation(len(dataset))
    shards = np.array_split(idx, num_clients)
    return [Subset(dataset, s.tolist()) for s in shards]


client_datasets = split_iid(train_set, NUM_CLIENTS)
print(f"{NUM_CLIENTS} clients, {len(client_datasets[0])} training samples each")


# ------------------------------------------------------------------
# 3. MODEL (same CNN as week 2)
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
# 4. CLIENT SIDE: local training
#    Takes the global model, trains a COPY on local data, returns the
#    copy's weights. The global model itself is never touched here.
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
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model.state_dict()


# ------------------------------------------------------------------
# 5. SERVER SIDE: FedAvg = average every weight tensor across clients
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 6. THE FEDERATED LEARNING LOOP
# ------------------------------------------------------------------
global_model = SimpleCNN()
accuracy_history = []

for rnd in range(1, NUM_ROUNDS + 1):
    # steps 1-3: broadcast, local training, collect updates
    client_states = [local_train(global_model, ds) for ds in client_datasets]

    # step 4: server aggregates
    global_model.load_state_dict(fedavg(client_states))

    # step 5: evaluate
    acc = evaluate(global_model, test_loader)
    accuracy_history.append(acc)
    print(f"Round {rnd}/{NUM_ROUNDS} - global test accuracy: {acc:.4f}")

# ------------------------------------------------------------------
# Save result + plot
# ------------------------------------------------------------------
plt.figure()
plt.plot(range(1, NUM_ROUNDS + 1), accuracy_history, marker="o")
plt.xlabel("Communication round")
plt.ylabel("Global test accuracy")
plt.title(f"FedAvg: {NUM_CLIENTS} clients, IID, no attackers")
plt.grid(True)
plt.savefig("fedavg_accuracy.png", dpi=150, bbox_inches="tight")
print("Saved: fedavg_accuracy.png")