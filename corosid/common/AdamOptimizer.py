import numpy as np
from tqdm import tqdm


class AdamOptimizer:
    def __init__(self,
                 forward_and_gradient,
                 x0,
                 num_iter,
                 alpha,
                 beta1,
                 beta2,
                 eps=1e-8,
                 callback=None):
        self.forward_and_gradient = forward_and_gradient
        self.x = x0
        self.num_iter = num_iter
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.callback = callback

        if self.callback is None:
            def callback(optimizer):
                pass

            self.callback = callback

        self.Js = np.zeros(num_iter)  # Cost function values
        self.m = np.zeros(x0.shape)
        self.v = np.zeros(x0.shape)

    def iterate(self, k):
        self.Js[k], g = self.forward_and_gradient(self.x)

        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (np.abs(g) ** 2)
        m_hat = self.m / (1.0 - self.beta1 ** (k + 1))
        v_hat = self.v / (1.0 - self.beta2 ** (k + 1))
        update = self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)
        self.x = self.x - update

        self.callback(self)

        return self.Js[k], self.x

    def optimize(self):
        for k in tqdm(range(self.num_iter)):
            self.iterate(k)

        return self.Js, self.x