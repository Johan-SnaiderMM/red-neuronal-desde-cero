/**
 * Convierte un dibujo hecho con el dedo o el mouse en una imagen con el mismo formato que MNIST, que es con lo
 * que se entrenó la red: dígito blanco sobre fondo negro, de 28×28 píxeles, escalado para caber en una caja de
 * 20×20 y centrado por su centro de masa. Sin este paso la red acierta mucho menos.
 */

export const LADO = 28;
export const CAJA = 20;
/** Intensidad mínima (0-1) para considerar que un píxel del lienzo tiene tinta. */
export const UMBRAL_TINTA = 0.08;

export interface Caja {
  izquierda: number;
  arriba: number;
  ancho: number;
  alto: number;
}

/** Rectángulo mínimo que contiene la tinta, o null si el lienzo está vacío. */
export function cajaDelTrazo(tinta: Float32Array, ancho: number, alto: number): Caja | null {
  let minX = ancho,
    minY = alto,
    maxX = -1,
    maxY = -1;
  for (let y = 0; y < alto; y++) {
    for (let x = 0; x < ancho; x++) {
      if (tinta[y * ancho + x] > UMBRAL_TINTA) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;
  return { izquierda: minX, arriba: minY, ancho: maxX - minX + 1, alto: maxY - minY + 1 };
}

/** Reduce (o amplía) un recorte de la imagen a destAncho×destAlto promediando el área que cubre cada píxel nuevo. */
export function remuestrearPorArea(
  origen: Float32Array,
  anchoOrigen: number,
  caja: Caja,
  destAncho: number,
  destAlto: number,
): Float32Array {
  const destino = new Float32Array(destAncho * destAlto);
  const altoOrigen = origen.length / anchoOrigen;
  const pasoX = caja.ancho / destAncho;
  const pasoY = caja.alto / destAlto;
  for (let dy = 0; dy < destAlto; dy++) {
    const y0 = caja.arriba + dy * pasoY;
    const y1 = y0 + pasoY;
    for (let dx = 0; dx < destAncho; dx++) {
      const x0 = caja.izquierda + dx * pasoX;
      const x1 = x0 + pasoX;
      let suma = 0;
      for (let sy = Math.floor(y0); sy < Math.ceil(y1); sy++) {
        // Por redondeo de decimales, `ceil(y1)` puede pasarse una fila (o columna) del borde de la imagen: lo de afuera es fondo.
        if (sy >= altoOrigen) continue;
        const pesoY = Math.min(sy + 1, y1) - Math.max(sy, y0);
        for (let sx = Math.floor(x0); sx < Math.ceil(x1); sx++) {
          if (sx >= anchoOrigen) continue;
          const pesoX = Math.min(sx + 1, x1) - Math.max(sx, x0);
          suma += origen[sy * anchoOrigen + sx] * pesoX * pesoY;
        }
      }
      destino[dy * destAncho + dx] = suma / (pasoX * pasoY);
    }
  }
  return destino;
}

/** Centro de masa (en coordenadas de índice de píxel) o null si no hay tinta. */
export function centroDeMasa(imagen: Float32Array, ancho: number, alto: number): { x: number; y: number } | null {
  let masa = 0,
    sx = 0,
    sy = 0;
  for (let y = 0; y < alto; y++) {
    for (let x = 0; x < ancho; x++) {
      const v = imagen[y * ancho + x];
      masa += v;
      sx += v * x;
      sy += v * y;
    }
  }
  return masa > 0 ? { x: sx / masa, y: sy / masa } : null;
}

/**
 * `tinta`: intensidades 0-1 del lienzo (1 = trazo), de `ancho`×`alto`. Devuelve 784 valores 0-1 o null si no hay trazo.
 */
export function prepararDigito(tinta: Float32Array, ancho: number, alto: number): Float32Array | null {
  const caja = cajaDelTrazo(tinta, ancho, alto);
  if (!caja) return null;

  // 1) Escalar para que el lado mayor mida CAJA (20 px) conservando la proporción.
  const escala = CAJA / Math.max(caja.ancho, caja.alto);
  const dw = Math.max(1, Math.round(caja.ancho * escala));
  const dh = Math.max(1, Math.round(caja.alto * escala));
  const reducida = remuestrearPorArea(tinta, ancho, caja, dw, dh);

  // 2) Pegarla en el centro del cuadro de 28×28.
  const salida = new Float32Array(LADO * LADO);
  const pegar = (desdeX: number, desdeY: number) => {
    salida.fill(0);
    for (let y = 0; y < dh; y++) {
      for (let x = 0; x < dw; x++) {
        const px = x + desdeX;
        const py = y + desdeY;
        if (px >= 0 && px < LADO && py >= 0 && py < LADO) salida[py * LADO + px] = reducida[y * dw + x];
      }
    }
  };
  pegar(Math.floor((LADO - dw) / 2), Math.floor((LADO - dh) / 2));

  // 3) Mover la imagen para que su centro de masa caiga en el centro del cuadro (13.5, 13.5 en índices), como en MNIST.
  const centro = centroDeMasa(salida, LADO, LADO);
  if (centro) {
    const ajusteX = Math.round((LADO - 1) / 2 - centro.x);
    const ajusteY = Math.round((LADO - 1) / 2 - centro.y);
    pegar(Math.floor((LADO - dw) / 2) + ajusteX, Math.floor((LADO - dh) / 2) + ajusteY);
  }
  return salida;
}
