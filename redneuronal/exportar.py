"""Exporta los pesos de la red a un JSON que el navegador entiende (float32 en base64, little-endian)."""

from __future__ import annotations

import base64

import numpy as np

from .red import RedNeuronal


def _a_base64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def _de_base64(texto: str, forma) -> np.ndarray:
    return np.frombuffer(base64.b64decode(texto), dtype="<f4").reshape(forma).copy()


def red_a_diccionario(red: RedNeuronal, **metadatos) -> dict:
    """Pesos W de cada capa en orden fila-mayor con forma (entrada, salida), más los sesgos b."""
    capas = []
    for capa in red._densas():
        capas.append(
            {"entrada": int(capa.W.shape[0]), "salida": int(capa.W.shape[1]), "W": _a_base64(capa.W), "b": _a_base64(capa.b)}
        )
    return {"formato": 1, "arquitectura": red.tamanos, "activacion": "relu", "capas": capas, **metadatos}


def diccionario_a_red(d: dict) -> RedNeuronal:
    """Operación inversa (se usa en las pruebas para comprobar que no se pierde nada)."""
    red = RedNeuronal(d["arquitectura"])
    for capa, datos in zip(red._densas(), d["capas"]):
        capa.W[...] = _de_base64(datos["W"], (datos["entrada"], datos["salida"]))
        capa.b[...] = _de_base64(datos["b"], (datos["salida"],))
    return red
