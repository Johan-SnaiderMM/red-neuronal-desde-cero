"""Exporta el modelo entrenado a la demo web: pesos (web/public/modelo.json) y ejemplos de prueba para verificar paridad.

El navegador hace la MISMA operación que NumPy (producto de matrices + ReLU). Para comprobar que no cambia nada
al cruzar de Python a TypeScript, se guardan algunas imágenes de prueba con los logits que calculó NumPy;
una prueba de Vitest los compara contra la implementación en TypeScript.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redneuronal import RedNeuronal  # noqa: E402
from redneuronal.datos import aplanar, cargar_mnist  # noqa: E402
from redneuronal.exportar import red_a_diccionario  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modelo", default="resultados/modelo_numpy_con_aumento.npz")
    ap.add_argument("--metricas", default="resultados/metricas_numpy_con_aumento.json")
    ap.add_argument("--datos", default="data")
    ap.add_argument("--web", default="web")
    a = ap.parse_args()

    red = RedNeuronal.cargar(a.modelo)
    metricas = json.loads(Path(a.metricas).read_text(encoding="utf-8"))
    web = Path(a.web)

    modelo = red_a_diccionario(
        red,
        parametros=red.num_parametros(),
        precision_prueba=metricas["precision_prueba"],
        entrenado_con="MNIST (55 000 imágenes de entrenamiento)",
        aumento_de_datos=metricas["aumento_de_datos"],
        fecha=date.today().isoformat(),
    )
    destino = web / "public" / "modelo.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(modelo, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{destino}  ({destino.stat().st_size / 1024:.0f} KB, {red.num_parametros():,} parámetros)")

    # Un ejemplo de cada dígito de la prueba (los primeros que aparecen) + los logits que calcula NumPy.
    _, _, x_prueba, y_prueba = cargar_mnist(a.datos)
    indices = [int(np.flatnonzero(y_prueba == d)[0]) for d in range(10)]
    pixeles = x_prueba[indices]
    logits = red.logits(aplanar(pixeles))
    ejemplos = [
        {"etiqueta": int(y_prueba[i]), "pixeles": p.reshape(-1).astype(int).tolist(), "logits": [round(float(v), 5) for v in fila]}
        for i, p, fila in zip(indices, pixeles, logits)
    ]
    fixtures = web / "src" / "lib" / "__fixtures__" / "ejemplos.json"
    fixtures.parent.mkdir(parents=True, exist_ok=True)
    fixtures.write_text(json.dumps(ejemplos, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{fixtures}  ({len(ejemplos)} ejemplos)")


if __name__ == "__main__":
    main()
