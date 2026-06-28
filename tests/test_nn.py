import torch
import torch.nn as nn
import torch.nn.functional as F

from nanograd.nn import Embedding, LayerNorm, Linear, Module
from nanograd.optim import SGD, Adam, AdamW


# --- Linear ---


def test_linear_forward_shape():
    lin = Linear(4, 8)
    x = torch.randn(3, 4)
    out = lin(x)
    assert out.shape == (3, 8)


def test_linear_grad_flows():
    lin = Linear(4, 8)
    x = torch.randn(3, 4)
    loss = lin(x).sum()
    loss.backward()
    assert lin.weight.grad is not None
    assert lin.bias.grad is not None


# --- LayerNorm ---


def test_layernorm_normalizes():
    ln = LayerNorm(8)
    x = torch.randn(3, 8) * 5 + 2
    out = ln(x)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(3), atol=1e-5)
    assert torch.allclose(out.std(dim=-1, unbiased=False), torch.ones(3), atol=1e-2)


def test_layernorm_grad_flows():
    ln = LayerNorm(8)
    x = torch.randn(3, 8)
    ln(x).sum().backward()
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None


def test_layernorm_x_grad_flows():
    ln = LayerNorm(8)
    x = torch.randn(3, 8, requires_grad=True)
    # sum(LayerNorm(x)) has zero grad through x by construction (normalized values sum to 0),
    # so use a squared loss to get non-zero gradient.
    (ln(x) ** 2).sum().backward()
    assert x.grad is not None
    assert not torch.all(x.grad == 0)


# --- Embedding ---


def test_embedding_lookup():
    emb = Embedding(10, 4)
    idx = torch.tensor([0, 2, 5])
    out = emb(idx)
    assert out.shape == (3, 4)
    assert torch.allclose(out[0], emb.weight[0])
    assert torch.allclose(out[1], emb.weight[2])


def test_embedding_grad_accumulates():
    emb = Embedding(10, 4)
    idx = torch.tensor([2, 3, 2])
    out = emb(idx)
    out.sum().backward()
    assert torch.allclose(emb.weight.grad[2], 2 * emb.weight.grad[3], atol=1e-6)


# --- Module ---


def test_module_parameters():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = Linear(2, 4)
            self.l2 = Linear(4, 2)

        def forward(self, x):
            return self.l2(F.relu(self.l1(x)))

    net = Net()
    params = list(net.parameters())
    assert len(params) == 4


def test_zero_grad():
    lin = Linear(4, 8)
    x = torch.randn(3, 4)
    lin(x).sum().backward()
    lin.zero_grad()
    assert lin.weight.grad is None or torch.all(lin.weight.grad == 0)


# --- Optimizers ---


def test_sgd_step():
    lin = Linear(4, 8)
    w_before = lin.weight.data.clone()
    x = torch.randn(3, 4)
    opt = SGD(lin.parameters(), lr=0.1, momentum=0.0)
    loss = lin(x).sum()
    loss.backward()
    opt.step()
    assert not torch.allclose(lin.weight.data, w_before)


def test_adam_step():
    lin = Linear(4, 8)
    w_before = lin.weight.data.clone()
    x = torch.randn(3, 4)
    opt = Adam(lin.parameters(), lr=0.01)
    loss = lin(x).sum()
    loss.backward()
    opt.step()
    assert not torch.allclose(lin.weight.data, w_before)


def test_adamw_step():
    lin = Linear(4, 8)
    w_before = lin.weight.data.clone()
    x = torch.randn(3, 4)
    opt = AdamW(lin.parameters(), lr=0.01, weight_decay=0.1)
    loss = lin(x).sum()
    loss.backward()
    opt.step()
    assert not torch.allclose(lin.weight.data, w_before)


# --- XOR convergence ---


def test_xor_convergence():
    torch.manual_seed(0)
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([0, 1, 1, 0])

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = Linear(2, 16)
            self.l2 = Linear(16, 2)

        def forward(self, x):
            return self.l2(F.relu(self.l1(x)))

    net = MLP()
    opt = Adam(net.parameters(), lr=0.01)

    for _ in range(1000):
        opt.zero_grad()
        loss = F.cross_entropy(net(X), y)
        loss.backward()
        opt.step()

    assert loss.item() < 0.01
