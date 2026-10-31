"""Entrena la red hecha a mano (solo NumPy) sobre MNIST y guarda modelo, métricas y gráficas."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redneuronal import Adam, RedNeuronal  # noqa: E402
from redneuronal.datos import aplanar, cargar_mnist, separar_validacion  # noqa: E402
from redneuronal.entrenar import entrenar, evaluar, matriz_confusion, metricas_por_clase  # noqa: E402
from redneuronal.informes import guardar_curvas, guardar_json, guardar_matriz  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datos", default="data", help="carpeta con mnist.npz")
    ap.add_argument("--salida", default="resultados", help="carpeta de resultados")
    ap.add_argument("--nombre", default="numpy", help="sufijo de los archivos de salida")
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--lote", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--aumentar", action="store_true", help="corrimientos y trazo más grueso durante el entrenamiento")
    a = ap.parse_args()

    x, y, x_prueba, y_prueba = cargar_mnist(a.datos)
    x, y, x_val, y_val = separar_validacion(aplanar(x), y, 5000, a.semilla)
    x_prueba = aplanar(x_prueba)
    print(f"entrenamiento {len(x)} · validación {len(x_val)} · prueba {len(x_prueba)}")

    red = RedNeuronal([784, 128, 64, 10], semilla=a.semilla)
    print(f"red 784-128-64-10 con {red.num_parametros():,} parámetros · aumento de datos: {'sí' if a.aumentar else 'no'}")
    inicio = time.perf_counter()
    historial = entrenar(red, Adam(red.parametros(), lr=a.lr), x, y, x_val, y_val, a.epocas, a.lote, a.semilla, aumentar=a.aumentar)
    total = time.perf_counter() - inicio

    # La prueba se mira UNA sola vez, al final.
    perdida_prueba, precision_prueba = evaluar(red, x_prueba, y_prueba)
    matriz = matriz_confusion(y_prueba, red.predecir(x_prueba))
    print(f"\nPRUEBA (10 000 imágenes que la red nunca vio): precisión {precision_prueba * 100:.2f} %  pérdida {perdida_prueba:.4f}  [{total:.0f} s]")

    salida = Path(a.salida)
    red.guardar(salida / f"modelo_{a.nombre}.npz")
    guardar_json(
        {
            "nombre": a.nombre,
            "arquitectura": red.tamanos,
            "parametros": red.num_parametros(),
            "optimizador": f"Adam(lr={a.lr})",
            "epocas": a.epocas,
            "lote": a.lote,
            "semilla": a.semilla,
            "aumento_de_datos": a.aumentar,
            "segundos_de_entrenamiento": round(total, 1),
            "precision_validacion_final": round(historial["precision_validacion"][-1], 4),
            "precision_prueba": round(precision_prueba, 4),
            "perdida_prueba": round(perdida_prueba, 4),
            "por_clase": metricas_por_clase(matriz),
            "matriz_confusion": matriz.tolist(),
            "historial": {k: [round(v, 5) for v in vs] for k, vs in historial.items()},
        },
        salida / f"metricas_{a.nombre}.json",
    )
    guardar_curvas(historial, salida / f"curvas_{a.nombre}.png", f"Red hecha a mano (NumPy) · {a.nombre}")
    guardar_matriz(matriz, [str(i) for i in range(10)], salida / f"matriz_confusion_{a.nombre}.png", f"Prueba MNIST · {a.nombre}")
    print(f"Resultados en {salida}/")


if __name__ == "__main__":
    main()
