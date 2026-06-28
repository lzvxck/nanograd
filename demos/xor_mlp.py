import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([0, 1, 1, 0])

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
)
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(1000):
    opt.zero_grad()
    loss = F.cross_entropy(model(X), y)
    loss.backward()
    opt.step()
    if (step + 1) % 200 == 0:
        print(f"step {step+1:4d}  loss {loss.item():.4f}")

final_loss = loss.item()
print(f"final loss: {final_loss:.4f}")
assert final_loss < 0.01, f"Did not converge: loss={final_loss}"
print("XOR converged.")
