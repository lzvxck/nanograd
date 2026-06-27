import numpy as np
import torch
import torch.nn.functional as F

from nanograd import Tensor

rng = np.random.default_rng(42)


def _arr(*shape):
    return rng.standard_normal(shape).astype(np.float32)


def _cmp(ng_grad, pt_grad):
    assert np.allclose(ng_grad, pt_grad.detach().numpy(), rtol=1e-4, atol=1e-6), (
        f"max diff: {np.abs(ng_grad - pt_grad.detach().numpy()).max():.6f}"
    )


def test_add_grad():
    a_np, b_np = _arr(3, 4), _arr(3, 4)
    a = Tensor(a_np.copy(), requires_grad=True)
    b = Tensor(b_np.copy(), requires_grad=True)
    a.add(b).sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    bt = torch.tensor(b_np, requires_grad=True)
    (at + bt).sum().backward()
    _cmp(a.grad, at.grad)
    _cmp(b.grad, bt.grad)


def test_sub_grad():
    a_np, b_np = _arr(3, 4), _arr(3, 4)
    a = Tensor(a_np.copy(), requires_grad=True)
    b = Tensor(b_np.copy(), requires_grad=True)
    a.sub(b).sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    bt = torch.tensor(b_np, requires_grad=True)
    (at - bt).sum().backward()
    _cmp(a.grad, at.grad)
    _cmp(b.grad, bt.grad)


def test_mul_grad():
    a_np, b_np = _arr(3, 4), _arr(3, 4)
    a = Tensor(a_np.copy(), requires_grad=True)
    b = Tensor(b_np.copy(), requires_grad=True)
    a.mul(b).sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    bt = torch.tensor(b_np, requires_grad=True)
    (at * bt).sum().backward()
    _cmp(a.grad, at.grad)
    _cmp(b.grad, bt.grad)


def test_matmul_grad():
    a_np, b_np = _arr(3, 4), _arr(4, 5)
    a = Tensor(a_np.copy(), requires_grad=True)
    b = Tensor(b_np.copy(), requires_grad=True)
    a.matmul(b).sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    bt = torch.tensor(b_np, requires_grad=True)
    (at @ bt).sum().backward()
    _cmp(a.grad, at.grad)
    _cmp(b.grad, bt.grad)


def test_pow_grad():
    a_np = np.abs(_arr(3, 4)) + 0.1
    a = Tensor(a_np.copy(), requires_grad=True)
    a.pow(3.0).sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    (at**3.0).sum().backward()
    _cmp(a.grad, at.grad)


def test_exp_grad():
    a_np = _arr(3, 4) * 0.5
    a = Tensor(a_np.copy(), requires_grad=True)
    a.exp().sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    at.exp().sum().backward()
    _cmp(a.grad, at.grad)


def test_log_grad():
    a_np = np.abs(_arr(3, 4)) + 0.1
    a = Tensor(a_np.copy(), requires_grad=True)
    a.log().sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    at.log().sum().backward()
    _cmp(a.grad, at.grad)


def test_sum_grad_no_axis():
    a_np = _arr(3, 4)
    a = Tensor(a_np.copy(), requires_grad=True)
    a.sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    at.sum().backward()
    _cmp(a.grad, at.grad)


def test_sum_grad_axis():
    a_np = _arr(3, 4)
    a = Tensor(a_np.copy(), requires_grad=True)
    a.sum(axis=1).sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    at.sum(dim=1).sum().backward()
    _cmp(a.grad, at.grad)


def test_mean_grad_no_axis():
    a_np = _arr(3, 4)
    a = Tensor(a_np.copy(), requires_grad=True)
    a.mean().backward()
    at = torch.tensor(a_np, requires_grad=True)
    at.mean().backward()
    _cmp(a.grad, at.grad)


def test_mean_grad_axis():
    a_np = _arr(3, 4)
    a = Tensor(a_np.copy(), requires_grad=True)
    a.mean(axis=1).sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    at.mean(dim=1).sum().backward()
    _cmp(a.grad, at.grad)


def test_reshape_grad():
    a_np = _arr(3, 4)
    a = Tensor(a_np.copy(), requires_grad=True)
    a.reshape(2, 6).sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    at.reshape(2, 6).sum().backward()
    _cmp(a.grad, at.grad)


def test_transpose_grad():
    a_np = _arr(3, 4)
    a = Tensor(a_np.copy(), requires_grad=True)
    a.transpose(1, 0).sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    at.transpose(1, 0).sum().backward()
    _cmp(a.grad, at.grad)


def test_relu_grad():
    a_np = _arr(3, 4)
    a = Tensor(a_np.copy(), requires_grad=True)
    a.relu().sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    at.relu().sum().backward()
    _cmp(a.grad, at.grad)


def test_sigmoid_grad():
    a_np = _arr(3, 4)
    a = Tensor(a_np.copy(), requires_grad=True)
    a.sigmoid().sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    at.sigmoid().sum().backward()
    _cmp(a.grad, at.grad)


def test_tanh_grad():
    a_np = _arr(3, 4)
    a = Tensor(a_np.copy(), requires_grad=True)
    a.tanh().sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    at.tanh().sum().backward()
    _cmp(a.grad, at.grad)


def test_softmax_grad():
    a_np = _arr(3, 4)
    a = Tensor(a_np.copy(), requires_grad=True)
    a.softmax(axis=-1).sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    F.softmax(at, dim=-1).sum().backward()
    _cmp(a.grad, at.grad)


def test_cross_entropy_grad():
    rng2 = np.random.default_rng(123)
    logits_np = rng2.standard_normal((5, 7)).astype(np.float32)
    targets_np = rng2.integers(0, 7, size=5)
    ng = Tensor(logits_np.copy(), requires_grad=True)
    ng.cross_entropy(targets_np).backward()
    pt = torch.tensor(logits_np, requires_grad=True)
    F.cross_entropy(pt, torch.tensor(targets_np, dtype=torch.long)).backward()
    _cmp(ng.grad, pt.grad)


def test_diamond_accumulation():
    x = Tensor(np.array([2.0], dtype=np.float32), requires_grad=True)
    y = x.mul(x)
    y.backward()
    assert np.isclose(x.grad[0], 4.0), f"expected 4.0, got {x.grad[0]}"


def test_softmax_stability():
    a_np = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    a = Tensor(a_np, requires_grad=True)
    out = a.softmax(axis=-1)
    assert not np.any(np.isnan(out.data)), "softmax produced nan on large inputs"
    out.sum().backward()
    assert not np.any(np.isnan(a.grad)), "softmax backward produced nan"


def test_add_broadcast_grad():
    a_np = _arr(3, 4)
    b_np = _arr(4)
    a = Tensor(a_np.copy(), requires_grad=True)
    b = Tensor(b_np.copy(), requires_grad=True)
    a.add(b).sum().backward()
    at = torch.tensor(a_np, requires_grad=True)
    bt = torch.tensor(b_np, requires_grad=True)
    (at + bt).sum().backward()
    _cmp(a.grad, at.grad)
    _cmp(b.grad, bt.grad)
