"""Bucle de entrenamiento por mini-lotes y métricas de evaluación."""

from __future__ import annotations

import time

import numpy as np

from .capas import PerdidaEntropiaCruzada
from .datos import aumentar_lote
from .red import RedNeuronal


def evaluar(red: RedNeuronal, x: np.ndarray, y: np.ndarray, tam_lote: int = 4096) -> tuple[float, float]:
    """Pérdida media y precisión (fracción de aciertos) sobre (x, y)."""
    perdida = PerdidaEntropiaCruzada()
    total, aciertos = 0.0, 0
    for i in range(0, len(x), tam_lote):
        logits = red.logits(x[i : i + tam_lote])
        etiquetas = y[i : i + tam_lote]
        total += perdida.forward(logits, etiquetas) * len(etiquetas)
        aciertos += int((logits.argmax(axis=1) == etiquetas).sum())
    return total / len(x), aciertos / len(x)


def matriz_confusion(y_real: np.ndarray, y_pred: np.ndarray, clases: int = 10) -> np.ndarray:
    """Filas = clase real, columnas = clase predicha."""
    m = np.zeros((clases, clases), dtype=np.int64)
    np.add.at(m, (y_real, y_pred), 1)
    return m


def metricas_por_clase(matriz: np.ndarray) -> list[dict]:
    """Precisión, exhaustividad (recall) y F1 de cada clase a partir de la matriz de confusión."""
    filas = []
    for c in range(matriz.shape[0]):
        tp = int(matriz[c, c])
        fp = int(matriz[:, c].sum() - tp)
        fn = int(matriz[c, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        filas.append({"clase": c, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "soporte": tp + fn})
    return filas


def entrenar(
    red: RedNeuronal,
    optimizador,
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epocas: int,
    tam_lote: int,
    semilla: int,
    aumentar: bool = False,
    registrar=print,
) -> dict:
    """Entrena `epocas` pasadas sobre los datos. Devuelve el historial por época."""
    rng = np.random.default_rng(semilla)
    historial = {"perdida_entrenamiento": [], "perdida_validacion": [], "precision_validacion": [], "segundos": []}
    n = len(x)
    for epoca in range(1, epocas + 1):
        t0 = time.perf_counter()
        orden = rng.permutation(n)
        suma = 0.0
        for i in range(0, n, tam_lote):
            idx = orden[i : i + tam_lote]
            lote = x[idx]
            if aumentar:
                lote = aumentar_lote(lote, rng)
            suma += red.paso(lote, y[idx]) * len(idx)
            optimizador.paso()
        perdida_val, precision_val = evaluar(red, x_val, y_val)
        seg = time.perf_counter() - t0
        historial["perdida_entrenamiento"].append(suma / n)
        historial["perdida_validacion"].append(perdida_val)
        historial["precision_validacion"].append(precision_val)
        historial["segundos"].append(seg)
        registrar(
            f"época {epoca:>2}/{epocas}  pérdida entrenamiento {suma / n:.4f}  "
            f"validación {perdida_val:.4f}  precisión validación {precision_val * 100:.2f} %  ({seg:.1f} s)"
        )
    return historial
