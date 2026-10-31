import pytest

from redneuronal.gradientes import error_relativo_maximo


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(decaimiento=0.01),
        dict(tamanos=(7, 8, 6, 5, 4)),
        dict(lote=1),
        dict(tamanos=(4, 3), lote=8),  # sin capas ocultas: regresión logística
    ],
)
def test_retropropagacion_coincide_con_diferencias_finitas(kwargs):
    """Si el backward escrito a mano está bien, el gradiente analítico y el numérico son el mismo número."""
    assert error_relativo_maximo(**kwargs) < 1e-6
