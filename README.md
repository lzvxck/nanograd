# nanograd

Minimal neural network library backed by [PyTorch](https://pytorch.org). Thin re-exports of `torch`, `torch.nn`, and `torch.optim` behind the `nanograd` namespace — same API shape, full autograd.

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
import torch

a = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
b = torch.tensor([[5., 6.], [7., 8.]], requires_grad=True)
loss = (a @ b).sum()
loss.backward()
print(a.grad)
```

## Training loop

```python
from nanograd.nn import Linear
from nanograd.optim import Adam
import torch.nn.functional as F

layer = Linear(4, 8)
opt = Adam(layer.parameters(), lr=1e-3)

for x_batch, y_batch in dataloader:
    opt.zero_grad()
    logits = layer(x_batch)
    loss = F.cross_entropy(logits, y_batch)
    loss.backward()
    opt.step()
```

## API

### `nanograd.nn`

| class | torch equivalent |
|-------|-----------------|
| `Module` | `torch.nn.Module` |
| `Linear(in, out)` | `torch.nn.Linear` |
| `LayerNorm(shape)` | `torch.nn.LayerNorm` |
| `Embedding(vocab, dim)` | `torch.nn.Embedding` |

### `nanograd.optim`

| class | torch equivalent |
|-------|-----------------|
| `SGD(params, lr, momentum)` | `torch.optim.SGD` |
| `Adam(params, lr, betas, eps)` | `torch.optim.Adam` |
| `AdamW(params, lr, ..., weight_decay)` | `torch.optim.AdamW` |

## Demos

```bash
python demos/xor_mlp.py          # 2-layer MLP on XOR; converges to loss < 0.01
python demos/char_lm.py          # character-level LM; loss drops over 500 steps
python demos/char_lm.py --text-file path/to/text.txt
```

## Tests

```bash
python -m pytest
```
