/**
 * Red neuronal de solo inferencia: la MISMA que se entrenó en NumPy (producto de matrices + ReLU + softmax).
 * Los pesos llegan como float32 en base64 (little-endian) dentro de `public/modelo.json`.
 */

export interface CapaDensa {
  entrada: number;
  salida: number;
  /** Forma (entrada, salida), fila a fila: el peso de la entrada i hacia la neurona j está en i·salida + j. */
  W: Float32Array;
  b: Float32Array;
}

export interface Modelo {
  arquitectura: number[];
  capas: CapaDensa[];
  parametros: number;
  precisionPrueba: number;
  entrenadoCon: string;
}

/** Forma del archivo modelo.json que genera scripts/exportar_web.py. */
export interface ModeloJSON {
  formato: number;
  arquitectura: number[];
  activacion: string;
  capas: { entrada: number; salida: number; W: string; b: string }[];
  parametros: number;
  precision_prueba: number;
  entrenado_con: string;
  aumento_de_datos: boolean;
  fecha: string;
}

function decodificarFloat32(base64: string, cantidad: number): Float32Array {
  const binario = atob(base64);
  if (binario.length !== cantidad * 4) {
    throw new Error(`Se esperaban ${cantidad * 4} bytes y llegaron ${binario.length}: el modelo está dañado.`);
  }
  const bytes = new Uint8Array(binario.length);
  for (let i = 0; i < binario.length; i++) bytes[i] = binario.charCodeAt(i);
  // DataView lee siempre en little-endian, sin depender de la máquina.
  const vista = new DataView(bytes.buffer);
  const salida = new Float32Array(cantidad);
  for (let i = 0; i < cantidad; i++) salida[i] = vista.getFloat32(i * 4, true);
  return salida;
}

export function cargarModelo(json: ModeloJSON): Modelo {
  if (json.formato !== 1 || json.activacion !== "relu") throw new Error("Formato de modelo no soportado.");
  const capas = json.capas.map((c) => ({
    entrada: c.entrada,
    salida: c.salida,
    W: decodificarFloat32(c.W, c.entrada * c.salida),
    b: decodificarFloat32(c.b, c.salida),
  }));
  return {
    arquitectura: json.arquitectura,
    capas,
    parametros: json.parametros,
    precisionPrueba: json.precision_prueba,
    entrenadoCon: json.entrenado_con,
  };
}

/** Pasada hacia adelante: devuelve los logits (puntajes sin normalizar) de cada clase. */
export function logits(modelo: Modelo, entrada: Float32Array): Float32Array {
  if (entrada.length !== modelo.capas[0].entrada) {
    throw new Error(`La red espera ${modelo.capas[0].entrada} valores y recibió ${entrada.length}.`);
  }
  let actual = entrada;
  modelo.capas.forEach((capa, indice) => {
    const salida = new Float32Array(capa.salida);
    salida.set(capa.b);
    // y_j = b_j + Σ_i x_i · W[i][j]
    for (let i = 0; i < capa.entrada; i++) {
      const xi = actual[i];
      if (xi === 0) continue; // una entrada en cero no aporta: en la imagen casi todo el fondo lo es
      const fila = i * capa.salida;
      for (let j = 0; j < capa.salida; j++) salida[j] += xi * capa.W[fila + j];
    }
    const esUltima = indice === modelo.capas.length - 1;
    if (!esUltima) for (let j = 0; j < salida.length; j++) if (salida[j] < 0) salida[j] = 0; // ReLU
    actual = salida;
  });
  return actual;
}

/** Softmax estable: se resta el máximo para que exp() no se desborde. */
export function softmax(puntajes: Float32Array): Float32Array {
  let maximo = -Infinity;
  for (const v of puntajes) if (v > maximo) maximo = v;
  const exps = new Float32Array(puntajes.length);
  let suma = 0;
  for (let i = 0; i < puntajes.length; i++) {
    exps[i] = Math.exp(puntajes[i] - maximo);
    suma += exps[i];
  }
  for (let i = 0; i < exps.length; i++) exps[i] /= suma;
  return exps;
}

export interface Clasificacion {
  probabilidades: Float32Array;
  /** Índice (dígito) con mayor probabilidad. */
  prediccion: number;
  confianza: number;
}

export function clasificar(modelo: Modelo, entrada: Float32Array): Clasificacion {
  const probabilidades = softmax(logits(modelo, entrada));
  let prediccion = 0;
  for (let i = 1; i < probabilidades.length; i++) if (probabilidades[i] > probabilidades[prediccion]) prediccion = i;
  return { probabilidades, prediccion, confianza: probabilidades[prediccion] };
}
