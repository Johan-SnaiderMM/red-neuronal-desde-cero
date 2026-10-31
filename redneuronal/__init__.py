"""Red neuronal desde cero: retropropagación escrita a mano con NumPy."""

from .optimizadores import SGD, Adam
from .red import RedNeuronal

__all__ = ["RedNeuronal", "SGD", "Adam"]
