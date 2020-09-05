import numpy as np


class Adam:

    def __init__(self, p=None, beta_1=0.9, beta_2=0.999, epsilon=1e-8, lr=1e-2, lr_decay_rate=0.9,
                 lr_decay_every=25):
        self.p = p
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_every = lr_decay_every
        self.m = [np.zeros(p_, dtype=float) for p_ in p]
        self.v = [np.zeros(p_, dtype=float) for p_ in p]
        self.t = [np.zeros(p_, dtype=float) for p_ in p]

        self.iter = 0
        self.lr_effective = lr

    def step(self, u, grad):
        if (self.iter + 1) % self.lr_decay_every == 0:
            self.lr_effective = self.lr_decay_rate * self.lr_effective

        self.m = [self.beta_1 * m_ + (1 - self.beta_1) * grad_ for (m_, grad_) in zip(self.m, grad)]
        self.v = [self.beta_2 * v_ + (1 - self.beta_2) * grad_ ** 2 for (v_, grad_) in zip(self.v, grad)]
        m_hat = [m_ / (1 - np.power(self.beta_1, t_ + 1)) for (m_, t_) in zip(self.m, self.t)]
        v_hat = [v_ / (1 - np.power(self.beta_2, t_ + 1)) for (v_, t_) in zip(self.v, self.t)]
        u_updated = [u_ - self.lr_effective * m_hat_ / (self.epsilon + np.sqrt(v_hat_))
                     for u_, m_hat_, v_hat_ in zip(u, m_hat, v_hat)]
        masks = [(u_ < 0) + (u_ > 1) for u_ in u_updated]
        for i, mask in enumerate(masks):
            if np.sum(mask) > 0:
                self.m[i][mask] = 0
                self.v[i][mask] = 0
                # u_updated[masks] = parameters[masks]
                u_updated[i][mask] = np.random.rand(np.sum(mask))
            new_indices = np.argsort(u_updated[i])
            u_updated[i] = u_updated[i][new_indices]
            self.m[i] = self.m[i][new_indices]
            self.v[i] = self.v[i][new_indices]
            self.t[i] = self.t[i][new_indices]
            self.t[i][~mask] = self.t[i][~mask] + 1
            self.t[i][mask] = 0
        self.iter = self.iter + 1
        return u_updated


class SGD:

    def __init__(self, lr=1e-2, lr_decay_rate=0.9, lr_decay_every=25):

        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_every = lr_decay_every
        self.iter = 0
        self.lr_effective = lr

    def step(self, u, grad):
        if (self.iter + 1) % self.lr_decay_every == 0:
            self.lr_effective = self.lr_decay_rate * self.lr_effective

        u_updated = [u_ - self.lr_effective * grad_ for (u_, grad_) in zip(u, grad)]

        masks = [(u_ < 0) + (u_ > 1) for u_ in u_updated]
        for i, mask in enumerate(masks):
            if np.sum(mask) > 0:
                # u_updated[i][mask] = u[i][mask]
                u_updated[i][mask] = np.random.rand(np.sum(mask))
            new_indices = np.argsort(u_updated[i])
            u_updated[i] = u_updated[i][new_indices]
        self.iter = self.iter + 1
        return u_updated
