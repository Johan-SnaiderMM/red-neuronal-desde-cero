"""Gráficas y archivos de resultados (matplotlib se importa solo si se piden las gráficas)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def guardar_json(datos: dict, ruta) -> None:
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Path(ruta).write_text(json.dumps(datos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def guardar_curvas(historial: dict, ruta, titulo: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epocas = np.arange(1, len(historial["perdida_entrenamiento"]) + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.plot(epocas, historial["perdida_entrenamiento"], label="entrenamiento")
    a.plot(epocas, historial["perdida_validacion"], label="validación")
    a.set(xlabel="época", ylabel="pérdida (entropía cruzada)", title="Pérdida")
    a.legend()
    b.plot(epocas, np.array(historial["precision_validacion"]) * 100, color="tab:green")
    b.set(xlabel="época", ylabel="precisión en validación (%)", title="Precisión")
    for ax in (a, b):
        ax.grid(alpha=0.3)
    fig.suptitle(titulo)
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def guardar_matriz(matriz: np.ndarray, etiquetas: list[str], ruta, titulo: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(etiquetas)
    fig, ax = plt.subplots(figsize=(0.75 * n + 2.2, 0.75 * n + 1.8))
    ax.imshow(matriz, cmap="Blues")
    ax.set(xticks=range(n), yticks=range(n), xticklabels=etiquetas, yticklabels=etiquetas, xlabel="predicha", ylabel="real", title=titulo)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    umbral = matriz.max() / 2 if matriz.max() else 1
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(matriz[i, j]), ha="center", va="center", color="white" if matriz[i, j] > umbral else "black", fontsize=9)
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
