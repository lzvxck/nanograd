import numpy as np

from nanograd import Tensor
from nanograd.nn import Embedding, LayerNorm, Linear, Module
from nanograd.optim import Adam, AdamW, SGD


def test_linear_forward_shape():
    lin = Linear(4, 8)
    x = Tensor(np.random.randn(3, 4).astype(np.float32))
    out = lin(x)
    assert out.shape == (3, 8)


def test_linear_grad_flows():
    lin = Linear(4, 8)
    x = Tensor(np.random.randn(3, 4).astype(np.float32))
    out = lin(x)
    out.sum().backward()
    assert lin.W.grad is not None
    assert not np.all(lin.W.grad == 0)
    assert lin.b.grad is not None
    assert not np.all(lin.b.grad == 0)


def test_layernorm_normalizes():
    ln = LayerNorm(8)
    ln.gamma.data[:] = 1.0
    ln.beta.data[:] = 0.0
    x = Tensor(np.random.randn(3, 8).astype(np.float32) * 5 + 10)
    out = ln(x)
    mean = out.data.mean(axis=-1)
    std = out.data.std(axis=-1)
    assert np.allclose(mean, 0.0, atol=1e-5), f"mean not approx 0: {mean}"
    assert np.allclose(std, 1.0, atol=1e-4), f"std not approx 1: {std}"


def test_layernorm_grad_flows():
    ln = LayerNorm(8)
    x = Tensor(np.random.randn(3, 8).astype(np.float32))
    out = ln(x)
    out.sum().backward()
    assert ln.gamma.grad is not None
    assert not np.all(ln.gamma.grad == 0)
    assert ln.beta.grad is not None
    assert not np.all(ln.beta.grad == 0)


def test_layernorm_x_grad_flows():
    ln = LayerNorm(8)
    x = Tensor(np.random.randn(3, 8).astype(np.float32), requires_grad=True)
    out = ln(x)
    out.sum().backward()
    assert x.grad is not None
    assert not np.all(x.grad == 0)


def test_embedding_lookup():
    emb = Embedding(10, 4)
    indices = np.array([0, 3, 0])
    out = emb(indices)
    assert out.shape == (3, 4)
    assert np.allclose(out.data[0], emb.weight.data[0])
    assert np.allclose(out.data[1], emb.weight.data[3])
    assert np.allclose(out.data[2], emb.weight.data[0])


def test_embedding_grad_accumulates():
    emb = Embedding(10, 4)
    indices = np.array([2, 2])
    out = emb(indices)
    out.sum().backward()
    assert np.allclose(emb.weight.grad[2], np.ones(4) * 2.0)


def test_module_parameters():
    class Net(Module):
        def __init__(self):
            self.lin1 = Linear(4, 8)
            self.lin2 = Linear(8, 2)

        def forward(self, x):
            return self.lin2(self.lin1(x).relu())

    net = Net()
    params = net.parameters()
    assert len(params) == 4


def test_zero_grad():
    lin = Linear(4, 8)
    x = Tensor(np.random.randn(3, 4).astype(np.float32))
    lin(x).sum().backward()
    lin.zero_grad()
    assert np.all(lin.W.grad == 0)
    assert np.all(lin.b.grad == 0)


def test_sgd_step():
    p = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
    p.grad = np.array([0.1, 0.2], dtype=np.float32)
    old_data = p.data.copy()
    opt = SGD([p], lr=0.5)
    opt.step()
    assert np.allclose(p.data, old_data - 0.5 * np.array([0.1, 0.2]))


def test_adam_step():
    p = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
    p.grad = np.array([0.1, 0.2], dtype=np.float32)
    old_data = p.data.copy()
    opt = Adam([p], lr=0.1)
    opt.step()
    assert np.all(p.data < old_data)


def test_adamw_step():
    p = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
    p.grad = np.zeros(2, dtype=np.float32)
    opt = AdamW([p], lr=0.1, weight_decay=0.1)
    opt.step()
    assert np.all(p.data < np.array([1.0, 2.0]))


def test_xor_convergence():
    np.random.seed(0)
    X = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    lin1 = Linear(2, 16)
    lin2 = Linear(16, 2)
    opt = Adam([*lin1.parameters(), *lin2.parameters()], lr=0.01)
    loss = None
    for _ in range(1000):
        logits = lin2(lin1(X).relu())
        loss = logits.cross_entropy(y)
        lin1.zero_grad()
        lin2.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.data) < 0.01, f"XOR did not converge: loss={loss.data:.4f}"
