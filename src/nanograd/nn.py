import numpy as np

from .tensor import Tensor


class Module:
    def parameters(self):
        params = []
        seen = set()
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad and id(v) not in seen:
                seen.add(id(v))
                params.append(v)
            elif isinstance(v, Module):
                for p in v.parameters():
                    if id(p) not in seen:
                        seen.add(id(p))
                        params.append(p)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, Tensor) and item.requires_grad and id(item) not in seen:
                        seen.add(id(item))
                        params.append(item)
                    elif isinstance(item, Module):
                        for p in item.parameters():
                            if id(p) not in seen:
                                seen.add(id(p))
                                params.append(p)
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        std = np.sqrt(2.0 / in_features)
        w_data = np.random.randn(in_features, out_features).astype(np.float32) * std
        self.W = Tensor(w_data, requires_grad=True)
        if bias:
            self.b = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
        else:
            self.b = None

    def forward(self, x):
        out = x.matmul(self.W)
        if self.b is not None:
            out = out.add(self.b)
        return out


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = np.float32(eps)
        self.gamma = Tensor(np.ones(normalized_shape, dtype=np.float32), requires_grad=True)
        self.beta = Tensor(np.zeros(normalized_shape, dtype=np.float32), requires_grad=True)

    def forward(self, x):
        axis = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        mean_data = x.data.mean(axis=axis, keepdims=True).astype(np.float32)
        var_data = x.data.var(axis=axis, keepdims=True).astype(np.float32)
        x_hat_data = ((x.data - mean_data) / np.sqrt(var_data + self.eps)).astype(np.float32)
        x_hat = Tensor(x_hat_data, requires_grad=False)
        return x_hat.mul(self.gamma).add(self.beta)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        scale = np.sqrt(1.0 / num_embeddings)
        w_data = np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * scale
        self.weight = Tensor(w_data, requires_grad=True)

    def forward(self, indices):
        indices_arr = np.asarray(indices, dtype=np.int64)
        out_data = self.weight.data[indices_arr].copy()
        out = Tensor(out_data, requires_grad=self.weight.requires_grad)
        weight_ref = self.weight
        indices_frozen = indices_arr.copy()

        def _back():
            if weight_ref.requires_grad:
                weight_ref._ensure_grad()
                # np.add.at handles repeated indices correctly (unbuffered scatter-add)
                np.add.at(weight_ref.grad, indices_frozen, out.grad)

        out._backward = _back
        out._prev = {self.weight} if self.weight.requires_grad else set()
        return out
