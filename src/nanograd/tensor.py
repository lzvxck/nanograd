import numpy as np


def _broadcast_grad(grad, target_shape):
    """Sum grad over axes that were broadcast to produce grad's shape from target_shape."""
    ndim = grad.ndim
    padded = (1,) * (ndim - len(target_shape)) + tuple(target_shape)
    axes = tuple(i for i, (g, t) in enumerate(zip(grad.shape, padded)) if t == 1)
    result = grad.sum(axis=axes, keepdims=True) if axes else grad
    return result.reshape(target_shape)


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set()

    def backward(self):
        topo, visited = [], set()

        def build(t):
            if t not in visited:
                visited.add(t)
                for p in t._prev:
                    build(p)
                topo.append(t)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)

    def _ensure_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def add(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        a_shape, b_shape = self.data.shape, other.data.shape

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _broadcast_grad(out.grad, a_shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += _broadcast_grad(out.grad, b_shape)

        out._backward = _back
        out._prev = {t for t in (self, other) if t.requires_grad}
        return out

    def sub(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        a_shape, b_shape = self.data.shape, other.data.shape

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _broadcast_grad(out.grad, a_shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += _broadcast_grad(-out.grad, b_shape)

        out._backward = _back
        out._prev = {t for t in (self, other) if t.requires_grad}
        return out

    def mul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        a_data, b_data = self.data.copy(), other.data.copy()
        a_shape, b_shape = self.data.shape, other.data.shape

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += _broadcast_grad(out.grad * b_data, a_shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += _broadcast_grad(out.grad * a_data, b_shape)

        out._backward = _back
        out._prev = {t for t in (self, other) if t.requires_grad}
        return out

    def matmul(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        a_data, b_data = self.data.copy(), other.data.copy()

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad @ b_data.T
            if other.requires_grad:
                other._ensure_grad()
                other.grad += a_data.T @ out.grad

        out._backward = _back
        out._prev = {t for t in (self, other) if t.requires_grad}
        return out

    def pow(self, n):
        out = Tensor(self.data ** n, requires_grad=self.requires_grad)
        a_data = self.data.copy()

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += n * a_data ** (n - 1) * out.grad

        out._backward = _back
        out._prev = {self} if self.requires_grad else set()
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)
        out_data = out.data.copy()

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out_data * out.grad

        out._backward = _back
        out._prev = {self} if self.requires_grad else set()
        return out

    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)
        a_data = self.data.copy()

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad / a_data

        out._backward = _back
        out._prev = {self} if self.requires_grad else set()
        return out

    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        a_shape = self.data.shape

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, a_shape).copy()

        out._backward = _back
        out._prev = {self} if self.requires_grad else set()
        return out

    def mean(self, axis=None, keepdims=False):
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        a_shape = self.data.shape
        if axis is None:
            n = self.data.size
        elif isinstance(axis, tuple):
            n = int(np.prod([self.data.shape[a] for a in axis]))
        else:
            n = self.data.shape[axis]

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, a_shape).copy() / n

        out._backward = _back
        out._prev = {self} if self.requires_grad else set()
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        a_shape = self.data.shape

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad.reshape(a_shape)

        out._backward = _back
        out._prev = {self} if self.requires_grad else set()
        return out

    def transpose(self, *axes):
        if not axes:
            axes = tuple(reversed(range(self.data.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad)
        inv_axes = tuple(np.argsort(axes))

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += np.transpose(out.grad, inv_axes)

        out._backward = _back
        out._prev = {self} if self.requires_grad else set()
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), requires_grad=self.requires_grad)
        a_data = self.data.copy()

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * (a_data > 0)

        out._backward = _back
        out._prev = {self} if self.requires_grad else set()
        return out

    def sigmoid(self):
        s = (1.0 / (1.0 + np.exp(-self.data))).astype(np.float32)
        out = Tensor(s, requires_grad=self.requires_grad)
        s_frozen = s.copy()

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * s_frozen * (1.0 - s_frozen)

        out._backward = _back
        out._prev = {self} if self.requires_grad else set()
        return out

    def tanh(self):
        t = np.tanh(self.data).astype(np.float32)
        out = Tensor(t, requires_grad=self.requires_grad)
        t_frozen = t.copy()

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * (1.0 - t_frozen**2)

        out._backward = _back
        out._prev = {self} if self.requires_grad else set()
        return out

    def softmax(self, axis=-1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(shifted).astype(np.float32)
        s = (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)
        out = Tensor(s, requires_grad=self.requires_grad)
        s_frozen = s.copy()

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                dout = out.grad
                self.grad += s_frozen * (dout - (dout * s_frozen).sum(axis=axis, keepdims=True))

        out._backward = _back
        out._prev = {self} if self.requires_grad else set()
        return out

    def cross_entropy(self, target):
        """Fused cross-entropy on raw logits. target is an integer array (no grad)."""
        logits = self.data
        target_arr = np.asarray(target, dtype=np.int64)
        n = target_arr.shape[0] if target_arr.ndim > 0 else 1
        shifted = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        log_sum_exp = np.log(e.sum(axis=-1))
        correct_logits = shifted[np.arange(n), target_arr]
        loss = float(-(correct_logits - log_sum_exp).mean())
        out = Tensor(np.float32(loss), requires_grad=self.requires_grad)
        logits_frozen = logits.copy()

        def _back():
            if self.requires_grad:
                self._ensure_grad()
                shifted2 = logits_frozen - logits_frozen.max(axis=-1, keepdims=True)
                e2 = np.exp(shifted2)
                probs = (e2 / e2.sum(axis=-1, keepdims=True)).astype(np.float32)
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(n), target_arr] = 1.0
                self.grad += out.grad * (probs - one_hot) / n

        out._backward = _back
        out._prev = {self} if self.requires_grad else set()
        return out

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return Tensor(np.float32(other)).sub(self)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.mul(other)

    def __matmul__(self, other):
        return self.matmul(other)

    def __pow__(self, n):
        return self.pow(n)

    def __neg__(self):
        return self.mul(Tensor(np.float32(-1.0)))

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
