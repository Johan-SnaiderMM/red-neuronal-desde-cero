import numpy as np
import pytest

from redneuronal import SGD, Adam, RedNeuronal
from redneuronal.capas import softmax
from redneuronal.entrenar import entrenar, evaluar


def tres_nubes(n=300, semilla=0):
    """Tres nubes gaussianas separables en 2D: un problema que cualquier red correcta aprende rápido."""
    rng = np.random.default_rng(semilla)
    centros = np.array([[0.0, 3.0], [3.0, -2.0], [-3.0, -2.0]])
    y = rng.integers(0, 3, size=n)
    x = centros[y] + rng.normal(scale=0.7, size=(n, 2))
    return x.astype(np.float32), y


def test_softmax_es_estable_con_logits_enormes():
    p = softmax(np.array([[1000.0, 1001.0, 999.0], [-1000.0, -1000.0, -1000.0]]))
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_la_salida_tiene_una_probabilidad_por_clase():
    red = RedNeuronal([6, 5, 4], semilla=1)
    p = red.probabilidades(np.random.default_rng(0).normal(size=(9, 6)).astype(np.float32))
    assert p.shape == (9, 4)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-5)


def test_cuenta_los_parametros():
    # 784·128 + 128 + 128·64 + 64 + 64·10 + 10
    assert RedNeuronal([784, 128, 64, 10]).num_parametros() == 109_386


def test_rechaza_una_arquitectura_sin_salida():
    with pytest.raises(ValueError):
        RedNeuronal([10])


@pytest.mark.parametrize("construir", [lambda p: Adam(p, lr=0.01), lambda p: SGD(p, lr=0.05, momentum=0.9)])
def test_aprende_un_problema_sencillo(construir):
    x, y = tres_nubes()
    x_val, y_val = tres_nubes(100, semilla=7)
    red = RedNeuronal([2, 16, 3], semilla=3)
    perdida_inicial, _ = evaluar(red, x_val, y_val)
    entrenar(red, construir(red.parametros()), x, y, x_val, y_val, epocas=25, tam_lote=32, semilla=0, registrar=lambda *_: None)
    perdida_final, precision = evaluar(red, x_val, y_val)
    assert perdida_final < perdida_inicial / 3
    assert precision > 0.95


def test_el_entrenamiento_es_reproducible_con_la_misma_semilla():
    x, y = tres_nubes()
    resultados = []
    for _ in range(2):
        red = RedNeuronal([2, 8, 3], semilla=5)
        entrenar(red, Adam(red.parametros(), lr=0.01), x, y, x, y, epocas=3, tam_lote=32, semilla=9, registrar=lambda *_: None)
        resultados.append(red.logits(x[:20]))
    np.testing.assert_array_equal(resultados[0], resultados[1])


def test_guardar_y_cargar_conserva_las_predicciones(tmp_path):
    x, y = tres_nubes()
    red = RedNeuronal([2, 8, 3], semilla=2)
    entrenar(red, Adam(red.parametros(), lr=0.01), x, y, x, y, epocas=2, tam_lote=32, semilla=0, registrar=lambda *_: None)
    ruta = tmp_path / "modelo.npz"
    red.guardar(ruta)
    otra = RedNeuronal.cargar(ruta)
    np.testing.assert_array_equal(red.logits(x), otra.logits(x))
