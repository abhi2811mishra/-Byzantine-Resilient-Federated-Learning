"""
week1_tensors.py
Goal: understand what a tensor is and the handful of operations that
every later piece of this project (models, gradients, aggregation) is
built on top of.

Run: python3 week1_tensors.py
"""
import torch

# A tensor is just a multi-dimensional array (like a numpy array) that
# PyTorch can compute gradients through.
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("x =\n", x)
print("shape:", x.shape)          # torch.Size([2, 2])
print("dtype:", x.dtype)          # torch.float32

# Basic ops — these are what "aggregation" will boil down to later:
# FedAvg is literally torch.stack(...).mean(dim=0)
y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
print("\nx + y =\n", x + y)
print("elementwise mean of x,y =\n", torch.stack([x, y]).mean(dim=0))

# Flattening — every model's weights get turned into ONE flat vector
# later so FedAvg/Median/Krum can all operate on "a list of vectors"
# instead of "a list of weird nested weight structures".
flat = x.view(-1)
print("\nflattened x:", flat)

# Autograd — the mechanism that makes training possible.
w = torch.tensor(2.0, requires_grad=True)
loss = (w - 5) ** 2          # a toy "loss": distance from w to 5
loss.backward()              # computes d(loss)/dw
print("\nw =", w.item(), " loss =", loss.item(), " gradient dw =", w.grad.item())
# gradient should be 2*(w-5) = 2*(2-5) = -6

if __name__ == "__main__":
    print("\nAll basic tensor operations ran successfully.")