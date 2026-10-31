# Red neuronal desde cero

Una red neuronal que reconoce dígitos escritos a mano, con la **retropropagación escrita a mano usando solo NumPy** (sin PyTorch ni TensorFlow), y una demo web donde dibujas un número y la misma red lo reconoce **en tu navegador**, sin servidor.

**Demo:** https://red-neuronal-desde-cero.vercel.app · dibuja con el dedo o el mouse.

![Tarjeta del proyecto](web/public/icons/og.png)

Es un proyecto de aprendizaje: quería entender de verdad qué pasa dentro de una red neuronal antes de usar un framework. Por eso cada pieza está escrita a mano y comprobada.

## Resultados

Red 784-128-64-10 (109 386 parámetros), ReLU, softmax y entropía cruzada, optimizador Adam, 20 épocas, lotes de 128. MNIST: 55 000 imágenes para entrenar, 5 000 para vigilar el entrenamiento (validación) y 10 000 de prueba que se miran **una sola vez, al final**.

| Modelo | Precisión en la prueba (10 000 imágenes) | Pérdida |
|---|---|---|
| Red a mano (NumPy), sin aumento de datos | 97,58 % | 0,1124 |
| Misma red en PyTorch, mismos datos y optimizador | 97,76 % | 0,1048 |
| **Red a mano (NumPy), con aumento de datos** → modelo de la demo | **98,09 %** | 0,0578 |

- La red a mano y PyTorch quedan a menos de 0,2 puntos: es lo esperable si la retropropagación está bien. Fue una sola semilla (42), así que diferencias de ese tamaño son ruido entre corridas, no una conclusión.
- El **aumento de datos** (corrimientos de hasta 2 píxeles y, a veces, trazo más grueso) lo decidí antes de ver la prueba, pensando en que la demo recibe dibujos hechos con el dedo, no dígitos de MNIST.
- Es un perceptrón multicapa: cerca de 98 % es lo normal para esta arquitectura. Una red convolucional pasa de 99 %; no era el objetivo.
- Los dígitos con menor exhaustividad son el 8 (95,9 %) y el 5 (97,8 %); las confusiones más frecuentes son 5→3, 2→7 y 4→9 (ver `resultados/matriz_confusion_numpy_con_aumento.png`).

Todos los números salen de `resultados/metricas_*.json` y de los registros `resultados/log_*.txt`.

## Cómo funciona

Para una capa densa `y = x·W + b`, con `L` la pérdida, la regla de la cadena da:

```
dL/dW = xᵀ · dL/dy        dL/db = Σ_lote dL/dy        dL/dx = dL/dy · Wᵀ
```

- **ReLU**: pasa el gradiente solo donde la entrada fue positiva.
- **Softmax + entropía cruzada**: el gradiente respecto a los logits es `(p − y_onehot) / N`.
- **Inicialización de He** (varianza `2/entrada`) y **Adam** con corrección de sesgo, ambos escritos a mano en `redneuronal/`.

## Cómo sé que está bien

1. **Verificación numérica de gradientes** (`scripts/verificar_gradientes.py`): se compara el gradiente analítico con diferencias finitas centrales. En redes de varias formas el error relativo queda en el orden de **1e-10**. Detalle que me costó entender: con sesgos iniciales en 0, una neurona cuya entrada es toda cero cae exactamente en la esquina de la ReLU, donde la derivada no está definida, y la comparación falla aunque el `backward` esté bien; por eso la verificación usa sesgos pequeños no nulos.
2. **Pruebas automáticas** (`pytest`, 19 pruebas): gradientes, softmax estable, que la red aprenda un problema sencillo con Adam y con SGD, reproducibilidad por semilla, guardar y cargar, aumento de datos y métricas.
3. **Comparación con PyTorch** (`scripts/entrenar_pytorch.py`): la misma arquitectura, inicialización, optimizador y datos.
4. **Paridad Python ↔ navegador** (`web/src/lib/red.test.ts`, Vitest): la red en TypeScript da los mismos logits que NumPy sobre 10 imágenes de prueba (hasta 3 decimales) y elige la misma clase.
5. **Preprocesamiento de la demo** (`web/src/lib/preprocesar.test.ts`): el dibujo se encuadra en 20×20 y se centra por centro de masa, como MNIST; una prueba amplía dígitos de prueba, los desplaza en un lienzo grande y comprueba que la red los siga reconociendo.

## Cómo reproducirlo

Requisitos: Python 3.10 o superior y Node 20 o superior (para la demo).

```bash
python -m venv .venv && source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements-torch.txt                      # NumPy, matplotlib, pytest y, solo para descargar MNIST y comparar, PyTorch

python scripts/descargar_mnist.py                          # ≈ 11 MB, desde el espejo de PyTorch, con verificación MD5
python scripts/verificar_gradientes.py                     # gradiente analítico vs numérico
pytest                                                     # 19 pruebas, no necesitan MNIST

python scripts/entrenar_numpy.py --nombre numpy_sin_aumento
python scripts/entrenar_numpy.py --nombre numpy_con_aumento --aumentar
python scripts/entrenar_pytorch.py                         # comparación

python scripts/exportar_web.py                             # pesos → web/public/modelo.json
cd web && npm install && npm run dev                       # demo en http://localhost:3000
npm run verificar                                          # lint + tipos + pruebas de la demo
```

Cada entrenamiento tarda unos 4 a 7 minutos en un portátil sin GPU.

## Estructura

```
redneuronal/      capas.py (backward a mano) · red.py · optimizadores.py (SGD, Adam) · datos.py · entrenar.py · gradientes.py · exportar.py
scripts/          descargar_mnist · verificar_gradientes · entrenar_numpy · entrenar_pytorch · exportar_web
tests/            pytest
resultados/       métricas, curvas, matrices de confusión, registros y el modelo entrenado (.npz)
web/              demo en Next.js 16 + TypeScript + Tailwind; la red corre en el navegador (src/lib/red.ts)
```

## Límites

- Solo reconoce dígitos del 0 al 9: si dibujas otra cosa, responde igual con un dígito.
- Se entrenó con MNIST (dígitos centrados, un solo estilo de captura); con dibujos muy distintos del estilo MNIST puede fallar.
- La demo acierta mejor con trazos grandes y de un solo golpe.

## Datos y licencias

- Código: MIT.
- MNIST: Yann LeCun, Corinna Cortes y Christopher J.C. Burges, CC BY-SA 3.0. Se descarga con `scripts/descargar_mnist.py`; no está en el repositorio.

Hecho por [Johan Moreno](https://github.com/Johan-SnaiderMM), estudiante de Ingeniería de Sistemas.
