import numpy as np

from redneuronal import RedNeuronal
from redneuronal.datos import aumentar_lote, separar_validacion
from redneuronal.entrenar import matriz_confusion, metricas_por_clase
from redneuronal.exportar import diccionario_a_red, red_a_diccionario


def test_separar_validacion_no_mezcla_los_conjuntos():
    x = np.arange(100, dtype=np.float32).reshape(100, 1)
    y = np.arange(100) % 10
    x_ent, y_ent, x_val, y_val = separar_validacion(x, y, n_validacion=20, semilla=3)
    assert len(x_ent) == 80 and len(x_val) == 20
    assert set(x_ent.ravel()).isdisjoint(set(x_val.ravel()))
    np.testing.assert_array_equal(y_val, x_val.ravel().astype(int) % 10)  # etiquetas siguen a su imagen


def cuadrado_centrado():
    img = np.zeros((1, 28, 28), dtype=np.float32)
    img[0, 10:18, 10:18] = 1.0
    return img.reshape(1, -1)


def test_aumento_sin_cambios_devuelve_lo_mismo():
    lote = cuadrado_centrado()
    salida = aumentar_lote(lote, np.random.default_rng(0), desplazamiento=0, prob_grosor=0.0)
    np.testing.assert_array_equal(salida, lote)


def test_el_desplazamiento_conserva_la_tinta_y_el_rango():
    lote = np.repeat(cuadrado_centrado(), 50, axis=0)
    salida = aumentar_lote(lote, np.random.default_rng(1), desplazamiento=2, prob_grosor=0.0)
    assert salida.shape == lote.shape
    np.testing.assert_allclose(salida.sum(axis=1), lote.sum(axis=1))  # el cuadrado está lejos del borde
    assert salida.min() >= 0.0 and salida.max() <= 1.0
    assert (salida != lote).any()  # sí se movió alguna


def test_engrosar_nunca_quita_tinta():
    lote = cuadrado_centrado()
    salida = aumentar_lote(lote, np.random.default_rng(2), desplazamiento=0, prob_grosor=1.0)
    assert salida.sum() > lote.sum()
    assert (salida >= lote).all()


def test_matriz_de_confusion_y_metricas():
    real = np.array([0, 0, 1, 1, 2, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 2, 0])
    m = matriz_confusion(real, pred, clases=3)
    np.testing.assert_array_equal(m, [[1, 1, 0], [0, 2, 0], [1, 0, 2]])
    por_clase = metricas_por_clase(m)
    assert por_clase[1]["recall"] == 1.0 and por_clase[1]["precision"] == round(2 / 3, 4)
    assert por_clase[2]["soporte"] == 3


def test_exportar_a_diccionario_y_volver_no_pierde_nada():
    red = RedNeuronal([784, 128, 64, 10], semilla=4)
    x = np.random.default_rng(0).random((16, 784), dtype=np.float32)
    d = red_a_diccionario(red, extra="dato")
    assert d["arquitectura"] == [784, 128, 64, 10] and d["extra"] == "dato" and len(d["capas"]) == 3
    np.testing.assert_array_equal(red.logits(x), diccionario_a_red(d).logits(x))
