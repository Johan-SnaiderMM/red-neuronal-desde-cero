"""Verificación numérica de gradientes ("gradient check").

La retropropagación escrita a mano se comprueba contra diferencias finitas centrales:
    dL/dθ ≈ (L(θ + ε) − L(θ − ε)) / 2ε
Si los dos números coinciden en todos los parámetros, el `backward` está bien.
"""

from __future__ import annotations

import numpy as np

from .red import RedNeuronal


def error_relativo_maximo(tamanos=(7, 6, 5, 4), lote: int = 5, decaimiento: float = 0.0, semilla: int = 0, eps: float = 1e-6) -> float:
    """Mayor error relativo ‖analítico − numérico‖ / (‖analítico‖ + ‖numérico‖) entre los tensores de parámetros de la red."""
    red = RedNeuronal(tamanos, semilla=semilla, dtype=np.float64, decaimiento=decaimiento)
    rng = np.random.default_rng(semilla + 1)
    # Sesgos pequeños y no nulos: con sesgo 0, una neurona cuya entrada es toda cero tiene pre-activación EXACTA 0, que es
    # justo la esquina de la ReLU (derivada indefinida) y haría fallar la comparación aunque el backward esté bien.
    for capa in red._densas():
        capa.b[...] = rng.normal(scale=0.1, size=capa.b.shape)
    x = rng.normal(size=(lote, tamanos[0]))
    y = rng.integers(0, tamanos[-1], size=lote)

    red.paso(x, y)
    analiticos = [g.copy() for _, g in red.parametros()]

    peor = 0.0
    for (p, _), analitico in zip(red.parametros(), analiticos):
        numerico = np.zeros_like(p)
        for i in np.ndindex(*p.shape):
            original = p[i]
            p[i] = original + eps
            mas = red.paso(x, y)
            p[i] = original - eps
            menos = red.paso(x, y)
            p[i] = original
            numerico[i] = (mas - menos) / (2.0 * eps)
        # Error relativo por normas del tensor completo. Comparar entrada por entrada engaña cuando un gradiente es
        # exactamente 0 (neuronas ReLU muertas): el numérico trae ~1e-10 de ruido de redondeo y daría "error 1".
        escala = np.linalg.norm(analitico) + np.linalg.norm(numerico)
        if escala > 1e-9:
            peor = max(peor, float(np.linalg.norm(analitico - numerico) / escala))
    return peor
