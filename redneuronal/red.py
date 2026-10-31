"""Perceptrón multicapa (MLP) hecho solo con NumPy: capas densas + ReLU y una salida con softmax."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .capas import Densa, PerdidaEntropiaCruzada, ReLU, softmax


class RedNeuronal:
    """Red totalmente conectada. `tamanos=[784, 128, 64, 10]` son 784 entradas, dos capas ocultas y 10 clases."""

    def __init__(self, tamanos, semilla: int = 0, dtype=np.float32, decaimiento: float = 0.0):
        if len(tamanos) < 2:
            raise ValueError("Hacen falta al menos una capa de entrada y una de salida.")
        rng = np.random.default_rng(semilla)
        self.tamanos = [int(t) for t in tamanos]
        self.decaimiento = decaimiento  # regularización L2 sobre los pesos (no sobre los sesgos)
        self.capas = []
        for i, (entrada, salida) in enumerate(zip(self.tamanos[:-1], self.tamanos[1:])):
            self.capas.append(Densa(entrada, salida, rng, dtype))
            if i < len(self.tamanos) - 2:
                self.capas.append(ReLU())
        self.perdida = PerdidaEntropiaCruzada()

    # --- inferencia -----------------------------------------------------------------------------
    def logits(self, x: np.ndarray) -> np.ndarray:
        for capa in self.capas:
            x = capa.forward(x)
        return x

    def probabilidades(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.logits(x))

    def predecir(self, x: np.ndarray) -> np.ndarray:
        return self.logits(x).argmax(axis=1)

    # --- entrenamiento --------------------------------------------------------------------------
    def paso(self, x: np.ndarray, etiquetas: np.ndarray) -> float:
        """Forward + pérdida + backward. Deja los gradientes en cada capa y devuelve la pérdida del lote."""
        valor = self.perdida.forward(self.logits(x), etiquetas)
        grad = self.perdida.backward()
        for capa in reversed(self.capas):
            grad = capa.backward(grad)
        if self.decaimiento:
            for capa in self._densas():
                capa.dW += self.decaimiento * capa.W
                valor += 0.5 * self.decaimiento * float((capa.W**2).sum())
        return valor

    def _densas(self):
        return [c for c in self.capas if isinstance(c, Densa)]

    def parametros(self):
        return [par for capa in self.capas for par in capa.parametros()]

    def num_parametros(self) -> int:
        return int(sum(p.size for p, _ in self.parametros()))

    # --- guardar y cargar -----------------------------------------------------------------------
    def guardar(self, ruta) -> None:
        datos = {"tamanos": np.array(self.tamanos)}
        for i, capa in enumerate(self._densas()):
            datos[f"W{i}"] = capa.W
            datos[f"b{i}"] = capa.b
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        np.savez(ruta, **datos)

    @classmethod
    def cargar(cls, ruta) -> "RedNeuronal":
        with np.load(ruta) as d:
            red = cls(d["tamanos"].tolist())
            for i, capa in enumerate(red._densas()):
                capa.W[...] = d[f"W{i}"]
                capa.b[...] = d[f"b{i}"]
        return red
