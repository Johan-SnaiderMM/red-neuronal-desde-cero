# Red neuronal desde cero

Una red neuronal que reconoce dígitos escritos a mano, con la **retropropagación escrita a mano usando solo NumPy** (sin PyTorch ni TensorFlow), y una demo web donde escribes un número, de uno o de varios dígitos, y la misma red lo lee **en tu navegador**, sin servidor.

**Demo:** https://red-neuronal-desde-cero.vercel.app · escribe con el dedo o el mouse (un dígito o varios, separados entre sí).

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
6. **Lectura de varios dígitos** (`web/src/lib/segmentar.test.ts` y `numero.test.ts`): la red sigue leyendo un dígito a la vez; antes se parte el dibujo en trazos continuos (componentes conexos con 8 vecinos), se unen los que se solapan en horizontal (la raya de un 4) y se descartan las manchas pequeñas. Las pruebas cubren el orden de izquierda a derecha, los dígitos de varios trazos, el ruido, que un vecino dentro de la caja no contamine el recorte, que un solo dígito dé lo mismo que antes y que se avise cuando hay dígitos pegados. Además, `web/src/lib/secuencias.test.ts` mide la tubería completa (segmentación + red) con **secuencias sintéticas** (ver más abajo).

## Varios dígitos: qué se midió y qué no

La red lee un dígito a la vez; para leer un número hay que partir el dibujo antes. Para medir esa parte armé **600 secuencias sintéticas** de 2 a 6 dígitos con imágenes de la **prueba de MNIST** (las que la red nunca vio al entrenar), ampliadas de 3× a 9× según quepan, con altura al azar, en un lienzo de 840×280 y **sin tocarse**. Semilla fija, así que es reproducible (`npm test` dentro de `web/` con MNIST descargado).

| Trazo | Bien separadas | Dígitos leídos bien | Número completo bien |
|---|---|---|---|
| Como el de MNIST | 98,0 % | 94,8 % | 86,5 % |
| 1,3 veces más grueso (parecido al lápiz de la demo) | 98,0 % | 94,2 % | 84,3 % |
| 1,8 veces más grueso (prueba de estrés) | 99,7 % | 93,5 % | 76,8 % |

Con trazo como el de MNIST, el número completo acierta 95,0 % en secuencias de 2 dígitos, 87,5 % en 3 y 4, 81,7 % en 5 y 80,8 % en 6. Un dígito aislado acierta 98,1 % por esta tubería frente a 98,2 % de la red sola (2 000 imágenes de prueba al azar).

Cómo leer estas cifras:
- **Son sintéticas.** Miden segmentación + lectura con dígitos de MNIST puestos uno al lado del otro, no escritura real de personas con el dedo. No sirven para afirmar «reconoce números escritos a mano» con ese porcentaje.
- «Dígitos leídos bien» **no cuenta** como acierto un dígito marcado como «pegado», aunque la red lo hubiera adivinado. Esas falsas alarmas son ~1,4 % de los dígitos: MNIST trae dígitos muy achatados (un «5» de 20×14 píxeles, un «6» casi acostado) que superan la regla «más ancho que alto». En una prueba con 20 000 parejas de dígitos pegados, esa regla avisa en el 84 %.
- El resto de las pérdidas viene de trazos finos de MNIST que se parten al ampliar (un «1» fino queda en dos trozos) y de la propia red (≈ 2 %).
- Los dígitos pegados quedan fuera: aquí nunca se tocan.
- Falta una medición con escritura real hecha en la demo.

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
npm run verificar                                          # lint + tipos + pruebas de la demo (con MNIST descargado también corre la
                                                           # evaluación de secuencias, ≈ 45 s; sin MNIST esa parte se salta)
```

Cada entrenamiento tarda unos 4 a 7 minutos en un portátil sin GPU.

## Estructura

```
redneuronal/      capas.py (backward a mano) · red.py · optimizadores.py (SGD, Adam) · datos.py · entrenar.py · gradientes.py · exportar.py
scripts/          descargar_mnist · verificar_gradientes · entrenar_numpy · entrenar_pytorch · exportar_web
tests/            pytest
resultados/       métricas, curvas, matrices de confusión, registros y el modelo entrenado (.npz)
web/              demo en Next.js 16 + TypeScript + Tailwind; la red corre en el navegador (src/lib/red.ts); la lectura de varios dígitos está en src/lib/segmentar.ts y numero.ts
```

## Límites

- Solo reconoce dígitos del 0 al 9: si dibujas otra cosa, responde igual con un dígito.
- Se entrenó con MNIST (dígitos centrados, un solo estilo de captura); con dibujos muy distintos del estilo MNIST puede fallar.
- La red lee **un dígito por imagen**. En un número, los dígitos tienen que ir **separados**: si se tocan forman un solo trazo y el programa no puede partirlos. Avisa cuando la caja es mucho más ancha que alta, pero dos dígitos angostos pegados (por ejemplo dos «1») pueden pasar por uno solo.
- La segmentación une los trazos que se solapan en horizontal; dos dígitos muy inclinados cuyos trazos se solapen en esa dirección podrían fundirse en uno.
- Lee hasta unos 6 dígitos en pantallas grandes y unos 4 en celular; la precisión de 98,09 % es para un dígito aislado de MNIST, no para números completos escritos a mano (ver la tabla de arriba: con secuencias sintéticas el número completo acierta entre 80 y 95 % según el largo).
- La demo acierta mejor con trazos grandes y de un solo golpe.

## Datos y licencias

- Código: MIT.
- MNIST: Yann LeCun, Corinna Cortes y Christopher J.C. Burges, CC BY-SA 3.0. Se descarga con `scripts/descargar_mnist.py`; no está en el repositorio.

Hecho por [Johan Moreno](https://github.com/Johan-SnaiderMM), Ingeniero de Sistemas.
