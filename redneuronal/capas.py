"""Capas de una red neuronal con la retropropagación escrita a mano (solo NumPy).

Cada capa guarda en `forward` lo que necesita para poder calcular su `backward`.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax estable: se resta el máximo de cada fila para que exp() no se desborde."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Densa:
    """Capa totalmente conectada: y = x @ W + b."""

    def __init__(self, entrada: int, salida: int, rng: np.random.Generator, dtype=np.float32):
        # Inicialización de He: varianza 2/entrada, pensada para activaciones ReLU.
        self.W = rng.normal(0.0, np.sqrt(2.0 / entrada), size=(entrada, salida)).astype(dtype)
        self.b = np.zeros(salida, dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_salida: np.ndarray) -> np.ndarray:
        # Regla de la cadena para y = xW + b:
        #   dL/dW = xᵀ · dL/dy      dL/db = suma de dL/dy sobre el lote      dL/dx = dL/dy · Wᵀ
        self.dW[...] = self._x.T @ grad_salida
        self.db[...] = grad_salida.sum(axis=0)
        return grad_salida @ self.W.T

    def parametros(self):
        """Pares (parámetro, gradiente). Se modifican en sitio: el optimizador guarda estas mismas referencias."""
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """max(0, x). Su derivada es 1 donde la entrada fue positiva y 0 donde no."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._activas = x > 0
        return x * self._activas

    def backward(self, grad_salida: np.ndarray) -> np.ndarray:
        return grad_salida * self._activas

    def parametros(self):
        return []


class PerdidaEntropiaCruzada:
    """Softmax + entropía cruzada juntas: el gradiente respecto a los logits es (p − y_onehot) / N."""

    def forward(self, logits: np.ndarray, etiquetas: np.ndarray) -> float:
        self._p = softmax(logits)
        self._etiquetas = etiquetas
        n = logits.shape[0]
        p_correcta = self._p[np.arange(n), etiquetas]
        return float(-np.log(np.clip(p_correcta, 1e-12, None)).mean())

    def backward(self) -> np.ndarray:
        n = self._p.shape[0]
        g = self._p.copy()
        g[np.arange(n), self._etiquetas] -= 1.0
        return g / n
