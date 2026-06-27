import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from nanograd import Tensor
from nanograd.nn import Linear
from nanograd.optim import Adam


def main():
    np.random.seed(42)
    X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
    y = np.array([0, 1, 1, 0], dtype=np.int64)

    lin1 = Linear(2, 16)
    lin2 = Linear(16, 2)
    opt = Adam([*lin1.parameters(), *lin2.parameters()], lr=0.01)

    loss = None
    for step in range(1, 1001):
        logits = lin2(lin1(X).relu())
        loss = logits.cross_entropy(y)
        lin1.zero_grad()
        lin2.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step:4d}  loss={loss.data:.6f}")

    assert float(loss.data) < 0.1, f"XOR did not converge: loss={loss.data:.4f}"
    print("XOR converged!")


if __name__ == "__main__":
    main()
