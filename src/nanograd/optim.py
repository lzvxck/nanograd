import numpy as np


class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = list(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocities):
            if p.grad is None:
                continue
            v[:] = self.momentum * v + p.grad
            p.data = (p.data - self.lr * v).astype(np.float32)


class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            m[:] = self.beta1 * m + (1.0 - self.beta1) * p.grad
            v[:] = self.beta2 * v + (1.0 - self.beta2) * p.grad**2
            m_hat = m / (1.0 - self.beta1**self.t)
            v_hat = v / (1.0 - self.beta2**self.t)
            p.data = (p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(np.float32)


class AdamW:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = float(lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            m[:] = self.beta1 * m + (1.0 - self.beta1) * p.grad
            v[:] = self.beta2 * v + (1.0 - self.beta2) * p.grad**2
            m_hat = m / (1.0 - self.beta1**self.t)
            v_hat = v / (1.0 - self.beta2**self.t)
            p.data = (
                p.data - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)
            ).astype(np.float32)
