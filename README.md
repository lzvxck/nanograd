# nanograd

Minimal reverse-mode autodiff library in pure Python (NumPy only). Pedagogically motivated — small enough to read in an afternoon, correct enough to train real networks.

## Install

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash
# .venv\Scripts\Activate.ps1   # PowerShell

pip install -e ".[dev]"
```

## Quick start

```python
from nanograd import Tensor
import numpy as np

a = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
b = Tensor([[5., 6.], [7., 8.]], requires_grad=True)
loss = (a.matmul(b)).sum()
loss.backward()
print(a.grad)  # dL/dA = ones @ B.T
```

## Training loop

```python
from nanograd.nn import Linear
from nanograd.optim import Adam

layer = Linear(4, 8)
opt = Adam(layer.parameters(), lr=1e-3)

for x_batch, y_batch in dataloader:
    logits = layer(x_batch)
    loss = logits.cross_entropy(y_batch)
    layer.zero_grad()
    loss.backward()
    opt.step()
```

## API

### `Tensor`

| op | notes |
|----|-------|
| `add`, `sub`, `mul`, `matmul` | broadcasting supported |
| `pow(n)`, `exp()`, `log()` | element-wise |
| `sum(axis)`, `mean(axis)` | reduction; tuple axis supported |
| `reshape(*shape)`, `transpose(*axes)` | view ops |
| `relu()`, `sigmoid()`, `tanh()` | activations |
| `softmax(axis)` | numerically stable |
| `cross_entropy(target)` | fused log-sum-exp; integer targets |
| `.backward()` | reverse-mode autodiff via topological sort |

### `nanograd.nn`

| class | description |
|-------|-------------|
| `Module` | base class; `parameters()`, `zero_grad()` |
| `Linear(in, out)` | `x @ W + b`; Kaiming init |
| `LayerNorm(shape)` | normalizes over last dims; grad flows through x |
| `Embedding(vocab, dim)` | integer index lookup; handles repeated indices |

### `nanograd.optim`

| class | notes |
|-------|-------|
| `SGD(params, lr, momentum)` | classical momentum |
| `Adam(params, lr, betas, eps)` | bias-corrected |
| `AdamW(params, lr, ..., weight_decay)` | decoupled weight decay |

## Demos

```bash
python demos/xor_mlp.py          # 2-layer MLP on XOR; converges to loss < 0.01
python demos/char_lm.py          # character-level LM; loss drops ~3.0 → 0.25
python demos/char_lm.py --text-file path/to/text.txt
```

## Tests

```bash
python -m pytest          # 34 tests; compares all 16 op gradients vs PyTorch
```

## Design

Each op returns a new `Tensor` and registers a `_backward` closure over the frozen inputs. `backward()` builds a topological order and calls each closure in reverse — the same algorithm as micrograd, scaled to arrays.

**Key invariants:**
- Gradients accumulate with `+=` — handles shared nodes (diamond DAG) automatically
- Parameter updates use reassignment (`p.data = p.data - lr * grad`), never in-place
- `Embedding` backward uses `np.add.at` to handle repeated indices correctly
- `cross_entropy` is a fused op on raw logits to avoid `log(softmax(x))` overflow
