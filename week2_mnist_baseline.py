"""
week2_mnist_baseline.py
Goal (Task 2 & 3 from the plan): train a real neural network on real MNIST
digits, save the test accuracy, and plot the training loss curve.

Same 6-step loop as week1_training_loop.py:
  zero_grad -> forward -> loss -> backward -> step
just with a real Dataset (MNIST) and a real model (a small CNN) instead
of toy points and a single linear layer.

Run: python3 week2_mnist_baseline.py
"""
import os
import gzip
import pickle
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ===================================================================
# 1. DATASET — MNIST: 60,000 training images + 10,000 test images of
#    handwritten digits 0-9, each 28x28 grayscale pixels.
# ===================================================================
MNIST_ROOT = "./mnist_data"
_FALLBACK_URL = ("https://raw.githubusercontent.com/mnielsen/"
                  "neural-networks-and-deep-learning/master/data/mnist.pkl.gz")


class _ArrayMNIST(Dataset):
    """Fallback dataset wrapper — only used if the standard torchvision
    download fails (e.g. your network blocks the official MNIST mirror).
    On most machines you'll never hit this path."""
    def __init__(self, images, labels, transform):
        self.images = (images.reshape(-1, 28, 28) * 255).astype(np.uint8)
        self.targets = torch.as_tensor(labels.astype(np.int64))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.targets[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def load_mnist():
    # ToTensor: converts a 28x28 image into a tensor and scales pixels
    #           from [0,255] to [0,1].
    # Normalize: rescales further using MNIST's known mean/std so the
    #           network trains faster and more stably.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    try:
        train_set = datasets.MNIST(MNIST_ROOT, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(MNIST_ROOT, train=False, download=True, transform=transform)
        return train_set, test_set
    except Exception as e:
        print(f"[data] Standard MNIST download failed ({e.__class__.__name__}); using fallback mirror.")
        os.makedirs(MNIST_ROOT, exist_ok=True)
        cache_path = os.path.join(MNIST_ROOT, "mnist_fallback.pkl.gz")
        if not os.path.exists(cache_path):
            urllib.request.urlretrieve(_FALLBACK_URL, cache_path)
        with gzip.open(cache_path, "rb") as f:
            train_data, valid_data, test_data = pickle.load(f, encoding="latin1")
        train_x = np.concatenate([train_data[0], valid_data[0]], axis=0)
        train_y = np.concatenate([train_data[1], valid_data[1]], axis=0)
        test_x, test_y = test_data
        return _ArrayMNIST(train_x, train_y, transform), _ArrayMNIST(test_x, test_y, transform)


train_set, test_set = load_mnist()
print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")


# ===================================================================
# 2. DATALOADER — batches + shuffles, exactly like week1.
# ===================================================================
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)


# ===================================================================
# 3. MODEL — a small CNN. Two conv layers (find visual patterns like
#    edges and curves) followed by two fully-connected layers (combine
#    those patterns into a digit classification).
# ===================================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 28x28 -> 28x28, 16 filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 14x14 -> 14x14, 32 filters
        self.pool = nn.MaxPool2d(2, 2)                            # halves height & width
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> [batch, 16, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))   # -> [batch, 32, 7, 7]
        x = x.view(x.size(0), -1)              # flatten for the FC layers
        x = F.relu(self.fc1(x))
        return self.fc2(x)


model = SimpleCNN()


# ===================================================================
# 4. LOSS FUNCTION — same as week1: CrossEntropyLoss for classification.
# ===================================================================
criterion = nn.CrossEntropyLoss()


# ===================================================================
# 5. OPTIMIZER — SGD with momentum (momentum helps convergence on real,
#    noisier data compared to the toy example).
# ===================================================================
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# ===================================================================
# helper: evaluate test accuracy
# ===================================================================
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


# ===================================================================
# 6. TRAINING LOOP — identical 5-line pattern to week1_training_loop.py:
#    zero_grad -> forward -> loss -> backward -> step
# ===================================================================
EPOCHS = 3
loss_history = []

for epoch in range(EPOCHS):
    model.train()
    epoch_losses = []
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    loss_history.append(avg_loss)
    test_acc = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - avg loss: {avg_loss:.4f} - test accuracy: {test_acc:.4f}")


# ===================================================================
# Task 3: save test accuracy + plot the training loss curve
# ===================================================================
final_acc = evaluate(model, test_loader)
print(f"\nFinal test accuracy: {final_acc:.4f}")

with open("mnist_baseline_result.txt", "w") as f:
    f.write(f"Final test accuracy: {final_acc:.4f}\n")
    f.write(f"Loss per epoch: {loss_history}\n")

plt.figure()
plt.plot(range(1, EPOCHS + 1), loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("MNIST CNN — Training Loss")
plt.grid(True)
plt.savefig("mnist_loss_curve.png", dpi=150, bbox_inches="tight")
print("Saved: mnist_baseline_result.txt, mnist_loss_curve.png"),