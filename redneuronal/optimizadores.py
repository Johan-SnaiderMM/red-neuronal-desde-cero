"""Optimizadores escritos a mano. Actualizan los parámetros EN SITIO usando los gradientes que dejó `backward`."""

from __future__ import annotations

import numpy as np


class SGD:
    """Descenso de gradiente con momentum: v ← μ·v − lr·g ;  p ← p + v."""

    def __init__(self, parametros, lr: float = 0.1, momentum: float = 0.9):
        self.parametros = parametros
        self.lr = lr
        self.momentum = momentum
        self._v = [np.zeros_like(p) for p, _ in parametros]

    def paso(self) -> None:
        for (p, g), v in zip(self.parametros, self._v):
            v *= self.momentum
            v -= self.lr * g
            p += v


class Adam:
    """Adam (Kingma y Ba, 2015): promedios móviles del gradiente y de su cuadrado, con corrección de sesgo."""

    def __init__(self, parametros, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.parametros = parametros
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self._m = [np.zeros_like(p) for p, _ in parametros]
        self._v = [np.zeros_like(p) for p, _ in parametros]
        self._t = 0

    def paso(self) -> None:
        self._t += 1
        corr1 = 1.0 - self.beta1**self._t
        corr2 = 1.0 - self.beta2**self._t
        for (p, g), m, v in zip(self.parametros, self._m, self._v):
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            p -= self.lr * (m / corr1) / (np.sqrt(v / corr2) + self.eps)
