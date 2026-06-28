import torch
import torch.nn.functional as F

from nanograd import Tensor


def make(*shape, val=1.0):
    return torch.full(shape, val, dtype=torch.float32, requires_grad=True)


def test_tensor_is_torch():
    t = Tensor([1.0, 2.0])
    assert isinstance(t, torch.Tensor)


def test_add():
    a, b = make(3, 4), make(3, 4, val=2.0)
    (a + b).sum().backward()
    assert a.grad is not None and a.grad.shape == (3, 4)
    assert b.grad is not None and b.grad.shape == (3, 4)


def test_sub():
    a, b = make(3, 4), make(3, 4, val=2.0)
    (a - b).sum().backward()
    assert a.grad is not None


def test_mul():
    a, b = make(3, 4), make(3, 4, val=2.0)
    (a * b).sum().backward()
    assert a.grad is not None and b.grad is not None


def test_matmul():
    a = make(3, 4)
    b = make(4, 5)
    (a @ b).sum().backward()
    assert a.grad.shape == (3, 4)
    assert b.grad.shape == (4, 5)


def test_pow():
    a = make(3, 4, val=2.0)
    (a**3).sum().backward()
    assert a.grad is not None


def test_exp():
    a = make(3, 4)
    a.exp().sum().backward()
    assert a.grad is not None


def test_log():
    a = make(3, 4, val=2.0)
    a.log().sum().backward()
    assert a.grad is not None


def test_sum_all():
    a = make(3, 4)
    a.sum().backward()
    assert a.grad.shape == (3, 4)


def test_sum_axis():
    a = make(3, 4)
    a.sum(dim=1).sum().backward()
    assert a.grad.shape == (3, 4)


def test_mean_all():
    a = make(3, 4)
    a.mean().backward()
    assert a.grad is not None


def test_mean_axis():
    a = make(3, 4)
    a.mean(dim=1).sum().backward()
    assert a.grad.shape == (3, 4)


def test_reshape():
    a = make(3, 4)
    a.reshape(12).sum().backward()
    assert a.grad.shape == (3, 4)


def test_transpose():
    a = make(3, 4)
    a.T.sum().backward()
    assert a.grad.shape == (3, 4)


def test_relu():
    a = make(3, 4, val=-0.5)
    F.relu(a).sum().backward()
    assert a.grad is not None


def test_sigmoid():
    a = make(3, 4)
    torch.sigmoid(a).sum().backward()
    assert a.grad is not None


def test_tanh():
    a = make(3, 4)
    torch.tanh(a).sum().backward()
    assert a.grad is not None


def test_softmax():
    a = make(3, 4)
    F.softmax(a, dim=-1).sum().backward()
    assert a.grad is not None


def test_cross_entropy():
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0])
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    assert logits.grad is not None and logits.grad.shape == (4, 3)


def test_diamond():
    x = torch.tensor([2.0], requires_grad=True)
    (x * x).backward()
    assert torch.allclose(x.grad, torch.tensor([4.0]))


def test_broadcast_add():
    a = make(3, 4)
    b = make(4)
    (a + b).sum().backward()
    assert a.grad.shape == (3, 4)
    assert b.grad.shape == (4,)
