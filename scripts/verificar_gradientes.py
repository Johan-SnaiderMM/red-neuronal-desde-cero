"""Comprueba la retropropagación escrita a mano contra diferencias finitas (ver redneuronal/gradientes.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redneuronal.gradientes import error_relativo_maximo  # noqa: E402


def main() -> None:
    casos = [
        ("sin regularización", dict(decaimiento=0.0)),
        ("con decaimiento L2 = 0.01", dict(decaimiento=0.01)),
        ("red más profunda (7-8-6-5-4)", dict(tamanos=(7, 8, 6, 5, 4))),
    ]
    peor = 0.0
    for nombre, kwargs in casos:
        error = error_relativo_maximo(**kwargs)
        peor = max(peor, error)
        print(f"{nombre:<32} error relativo máximo: {error:.2e}")
    ok = peor < 1e-6
    print("✔ los gradientes coinciden" if ok else "✘ los gradientes NO coinciden")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
