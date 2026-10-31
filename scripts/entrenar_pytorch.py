"""La MISMA red (784-128-64-10), el mismo optimizador y los mismos datos, pero con PyTorch.

Sirve de contraste: si la red hecha a mano está bien, debe llegar a una precisión casi igual a la de PyTorch.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redneuronal.datos import aplanar, cargar_mnist, separar_validacion  # noqa: E402
from redneuronal.entrenar import matriz_confusion, metricas_por_clase  # noqa: E402
from redneuronal.informes import guardar_json  # noqa: E402


def construir() -> nn.Sequential:
    modelo = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    for capa in modelo:
        if isinstance(capa, nn.Linear):  # misma inicialización de He que la versión a mano
            nn.init.kaiming_normal_(capa.weight, nonlinearity="relu")
            nn.init.zeros_(capa.bias)
    return modelo


@torch.no_grad()
def evaluar(modelo, x, y):
    logits = modelo(x)
    return float(nn.functional.cross_entropy(logits, y)), float((logits.argmax(1) == y).float().mean()), logits.argmax(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datos", default="data")
    ap.add_argument("--salida", default="resultados")
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--lote", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.semilla)
    x, y, x_prueba, y_prueba = cargar_mnist(a.datos)
    x, y, x_val, y_val = separar_validacion(aplanar(x), y, 5000, a.semilla)  # mismos 5 000 de validación que la versión a mano
    x, x_val, x_prueba = (torch.from_numpy(v) for v in (x, x_val, aplanar(x_prueba)))
    y, y_val, y_prueba = (torch.from_numpy(v.astype(np.int64)) for v in (y, y_val, y_prueba))

    modelo = construir()
    opt = torch.optim.Adam(modelo.parameters(), lr=a.lr)
    azar = torch.Generator().manual_seed(a.semilla)
    inicio = time.perf_counter()
    for epoca in range(1, a.epocas + 1):
        modelo.train()
        orden = torch.randperm(len(x), generator=azar)
        suma = 0.0
        for i in range(0, len(x), a.lote):
            idx = orden[i : i + a.lote]
            opt.zero_grad()
            perdida = nn.functional.cross_entropy(modelo(x[idx]), y[idx])
            perdida.backward()
            opt.step()
            suma += perdida.item() * len(idx)
        modelo.eval()
        perdida_val, precision_val, _ = evaluar(modelo, x_val, y_val)
        print(f"época {epoca:>2}/{a.epocas}  pérdida entrenamiento {suma / len(x):.4f}  validación {perdida_val:.4f}  precisión validación {precision_val * 100:.2f} %")
    total = time.perf_counter() - inicio

    perdida_prueba, precision_prueba, pred = evaluar(modelo, x_prueba, y_prueba)
    matriz = matriz_confusion(y_prueba.numpy(), pred.numpy())
    print(f"\nPRUEBA: precisión {precision_prueba * 100:.2f} %  pérdida {perdida_prueba:.4f}  [{total:.0f} s]")
    guardar_json(
        {
            "nombre": "pytorch",
            "arquitectura": [784, 128, 64, 10],
            "parametros": sum(p.numel() for p in modelo.parameters()),
            "optimizador": f"Adam(lr={a.lr})",
            "epocas": a.epocas,
            "lote": a.lote,
            "semilla": a.semilla,
            "aumento_de_datos": False,
            "segundos_de_entrenamiento": round(total, 1),
            "precision_prueba": round(precision_prueba, 4),
            "perdida_prueba": round(perdida_prueba, 4),
            "por_clase": metricas_por_clase(matriz),
            "torch": torch.__version__,
        },
        Path(a.salida) / "metricas_pytorch.json",
    )


if __name__ == "__main__":
    main()
