"""Carga de MNIST (descargado antes con scripts/descargar_mnist.py) y aumento de datos con NumPy."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def cargar_mnist(directorio) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (x_entrenamiento, y_entrenamiento, x_prueba, y_prueba): imágenes uint8 de 28×28 y etiquetas 0-9."""
    ruta = Path(directorio) / "mnist.npz"
    if not ruta.exists():
        raise FileNotFoundError(f"No encuentro {ruta}. Primero corre: python scripts/descargar_mnist.py --datos {directorio}")
    with np.load(ruta) as d:
        return d["x_entrenamiento"], d["y_entrenamiento"], d["x_prueba"], d["y_prueba"]


def aplanar(imagenes_uint8: np.ndarray) -> np.ndarray:
    """(N, 28, 28) uint8 → (N, 784) float32 en [0, 1]."""
    return (imagenes_uint8.reshape(len(imagenes_uint8), -1).astype(np.float32)) / 255.0


def separar_validacion(x: np.ndarray, y: np.ndarray, n_validacion: int = 5000, semilla: int = 0):
    """Aparta `n_validacion` ejemplos al azar (fijo por semilla) para vigilar el entrenamiento sin tocar la prueba."""
    orden = np.random.default_rng(semilla).permutation(len(x))
    val, ent = orden[:n_validacion], orden[n_validacion:]
    return x[ent], y[ent], x[val], y[val]


def aumentar_lote(lote: np.ndarray, rng: np.random.Generator, desplazamiento: int = 2, prob_grosor: float = 0.3) -> np.ndarray:
    """Variaciones baratas que se parecen a lo que dibuja una persona.

    Corre cada imagen hasta `desplazamiento` píxeles en x e y y, con probabilidad `prob_grosor`, engrosa el trazo
    un píxel (dilatación con los 4 vecinos). Recibe y devuelve (N, 784).
    """
    n = lote.shape[0]
    imgs = lote.reshape(n, 28, 28)
    m = desplazamiento
    relleno = np.pad(imgs, ((0, 0), (m, m), (m, m)))
    dx = rng.integers(-m, m + 1, size=n)
    dy = rng.integers(-m, m + 1, size=n)
    salida = np.empty_like(imgs)
    for i in range(n):
        salida[i] = relleno[i, m - dy[i] : m - dy[i] + 28, m - dx[i] : m - dx[i] + 28]
    gruesos = rng.random(n) < prob_grosor
    if gruesos.any():
        g = salida[gruesos]
        e = g.copy()
        e[:, 1:, :] = np.maximum(e[:, 1:, :], g[:, :-1, :])
        e[:, :-1, :] = np.maximum(e[:, :-1, :], g[:, 1:, :])
        e[:, :, 1:] = np.maximum(e[:, :, 1:], g[:, :, :-1])
        e[:, :, :-1] = np.maximum(e[:, :, :-1], g[:, :, 1:])
        salida[gruesos] = e
    return salida.reshape(n, -1)
