"""Descarga MNIST (≈ 11 MB) con torchvision y lo guarda como un solo archivo `mnist.npz`.

torchvision baja los 4 archivos del espejo de PyTorch (https://ossci-datasets.s3.amazonaws.com/mnist/) y comprueba su MD5.
MNIST: 70 000 dígitos escritos a mano de 28×28 píxeles (60 000 de entrenamiento y 10 000 de prueba).
Licencia: CC BY-SA 3.0. Autores: Yann LeCun, Corinna Cortes y Christopher J.C. Burges.
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default="data", help="carpeta donde guardar MNIST (por defecto ./data)")
    a = ap.parse_args()

    from torchvision.datasets import MNIST  # import tardío: el resto del proyecto no necesita torch

    carpeta = Path(a.datos)
    carpeta.mkdir(parents=True, exist_ok=True)
    entrenamiento = MNIST(carpeta / "crudo", train=True, download=True)
    prueba = MNIST(carpeta / "crudo", train=False, download=True)
    np.savez_compressed(
        carpeta / "mnist.npz",
        x_entrenamiento=entrenamiento.data.numpy(),
        y_entrenamiento=entrenamiento.targets.numpy().astype(np.uint8),
        x_prueba=prueba.data.numpy(),
        y_prueba=prueba.targets.numpy().astype(np.uint8),
    )
    print(f"Listo: {carpeta / 'mnist.npz'}  ({len(entrenamiento)} de entrenamiento, {len(prueba)} de prueba)")


if __name__ == "__main__":
    main()
