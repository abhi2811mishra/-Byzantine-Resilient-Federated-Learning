"""
week1_training_loop.py
Goal: understand every piece of a training loop on tiny, fake data
before pointing it at real MNIST. This is the exact same structure
Milestone 1 (centralized MNIST) and every FL client later will use.

Run: python3 week1_training_loop.py
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------
# 1. Dataset — tells PyTorch how many samples exist and how to fetch one.
#    Real MNIST works the same way; this fakes 100 samples of 2D points
#    that are trying to teach a model "is x[0] + x[1] > 0?"
# -----------------------------------------------------------------
class ToyDataset(Dataset):
    def __init__(self, n=100):
        torch.manual_seed(0)
        self.X = torch.randn(n, 2)
        self.y = (self.X.sum(dim=1) > 0).long()   # label: 0 or 1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------------------------------------------
# 2. DataLoader — batches the dataset and shuffles it each epoch.
# -----------------------------------------------------------------
dataset = ToyDataset(n=100)
loader = DataLoader(dataset, batch_size=10, shuffle=True)


# -----------------------------------------------------------------
# 3. Model — the smallest possible neural network: one linear layer.
#    Same idea as SimpleCNN later, just without conv layers.
# -----------------------------------------------------------------
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)   # 2 input features -> 2 class scores

    def forward(self, x):
        return self.fc(x)


model = TinyModel()

# -----------------------------------------------------------------
# 4. Loss function — measures how wrong the model's predictions are.
# -----------------------------------------------------------------
criterion = nn.CrossEntropyLoss()

# -----------------------------------------------------------------
# 5. Optimizer — decides how to update weights using the gradients
#    that .backward() computes.
# -----------------------------------------------------------------
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# -----------------------------------------------------------------
# 6. The training loop itself — this exact 5-line pattern
#    (zero_grad -> forward -> loss -> backward -> step) is what
#    every FL client will run locally, every round.
# -----------------------------------------------------------------
for epoch in range(5):
    epoch_loss = 0.0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()             # clear old gradients
        outputs = model(x_batch)          # forward pass
        loss = criterion(outputs, y_batch)  # how wrong were we?
        loss.backward()                   # compute gradients
        optimizer.step()                  # update weights
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}: avg loss = {epoch_loss/len(loader):.4f}")

# Check it actually learned something:
with torch.no_grad():
    preds = model(dataset.X).argmax(dim=1)
    acc = (preds == dataset.y).float().mean().item()
print(f"\nFinal training accuracy on toy data: {acc:.2%}")