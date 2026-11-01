/**
 * Evaluación de la lectura de varios dígitos con SECUENCIAS SINTÉTICAS armadas con las 10 000 imágenes de PRUEBA de MNIST
 * (la red nunca las vio al entrenar). No son números escritos a mano de verdad: son dígitos de MNIST ampliados y puestos
 * uno al lado del otro en un lienzo de 840×280, con tamaño y altura al azar, sin tocarse. Mide la tubería completa:
 * segmentación + preparación + red.
 *
 * Necesita los archivos de MNIST (`python scripts/descargar_mnist.py`, o los dos `t10k-*` en `data/crudo/MNIST/raw/`).
 * Sin ellos, la prueba se salta.
 */

import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import modeloJson from "../../public/modelo.json";
import { reconocerNumero } from "./numero";
import { cargarModelo, clasificar, type ModeloJSON } from "./red";

const RUTA_IMAGENES = fileURLToPath(new URL("../../../data/crudo/MNIST/raw/t10k-images-idx3-ubyte", import.meta.url));
const RUTA_ETIQUETAS = fileURLToPath(new URL("../../../data/crudo/MNIST/raw/t10k-labels-idx1-ubyte", import.meta.url));
const hayDatos = existsSync(RUTA_IMAGENES) && existsSync(RUTA_ETIQUETAS);

const ANCHO = 840;
const ALTO = 280;
const MARGEN = 8;
const SECUENCIAS = 600;
const SEMILLA = 20261101;

/** Generador pseudoaleatorio con semilla (mulberry32), para que la medición sea reproducible. */
function aleatorio(semilla: number) {
  let a = semilla >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function cargarMnist() {
  const bi = readFileSync(RUTA_IMAGENES);
  const bl = readFileSync(RUTA_ETIQUETAS);
  if (bi.readUInt32BE(0) !== 2051 || bl.readUInt32BE(0) !== 2049) throw new Error("Archivos de MNIST con formato inesperado.");
  const n = bi.readUInt32BE(4);
  if (bi.readUInt32BE(8) !== 28 || bi.readUInt32BE(12) !== 28 || bl.readUInt32BE(4) !== n) throw new Error("MNIST: tamaños inesperados.");
  const imagenes = Array.from({ length: n }, (_, i) => bi.subarray(16 + i * 784, 16 + (i + 1) * 784));
  const etiquetas = Array.from(bl.subarray(8, 8 + n));
  return { imagenes, etiquetas };
}

/** Caja (en celdas de 28×28) de la tinta de una imagen de MNIST. */
function cajaTinta(img: Uint8Array) {
  let x0 = 28,
    y0 = 28,
    x1 = -1,
    y1 = -1;
  for (let y = 0; y < 28; y++)
    for (let x = 0; x < 28; x++)
      if (img[y * 28 + x] > 20) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
  return { x0, y0, ancho: x1 - x0 + 1, alto: y1 - y0 + 1 };
}

/** Engrosa el trazo `r` píxeles por lado, solo dentro del rectángulo dado (máximo en una ventana cuadrada, en dos pasadas). */
function engrosarRegion(tinta: Float32Array, r: number, x0: number, y0: number, x1: number, y1: number) {
  if (r <= 0) return;
  x0 = Math.max(0, x0);
  y0 = Math.max(0, y0);
  x1 = Math.min(ANCHO - 1, x1);
  y1 = Math.min(ALTO - 1, y1);
  const horizontal = new Float32Array(tinta.length);
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++) {
      let m = 0;
      for (let k = Math.max(x0, x - r); k <= Math.min(x1, x + r); k++) m = Math.max(m, tinta[y * ANCHO + k]);
      horizontal[y * ANCHO + x] = m;
    }
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++) {
      let m = 0;
      for (let k = Math.max(y0, y - r); k <= Math.min(y1, y + r); k++) m = Math.max(m, horizontal[k * ANCHO + x]);
      tinta[y * ANCHO + x] = m;
    }
}

/** Grosor típico del trazo de MNIST, en celdas de 28×28 (a ojo, sobre los dígitos ampliados). */
const TRAZO_MNIST = 2.5;

/**
 * Pone cada dígito en su propio tramo del lienzo (ancho/n), ampliado de 3× a 9× según quepa y a una altura al azar.
 * Los tramos no se solapan, así que los dígitos nunca se tocan: esto mide la segmentación y la lectura, no el caso de pegados.
 */
function componer(imagenes: Uint8Array[], indices: number[], rand: () => number, multiplicador: number): Float32Array {
  const n = indices.length;
  const tramo = Math.floor(ANCHO / n);
  const sMax = Math.min(9, Math.floor((tramo - 2 * MARGEN) / 20));
  const sMin = Math.max(3, sMax - 3);
  const lienzo = new Float32Array(ANCHO * ALTO);
  indices.forEach((indice, i) => {
    const img = imagenes[indice];
    const caja = cajaTinta(img);
    let s = sMin + Math.floor(rand() * (sMax - sMin + 1));
    while (caja.ancho * s > tramo - 2 * MARGEN && s > 2) s--;
    const anchoPx = caja.ancho * s;
    const altoPx = caja.alto * s;
    const x0 = i * tramo + MARGEN + Math.floor(rand() * (tramo - 2 * MARGEN - anchoPx + 1));
    const centrado = (ALTO - altoPx) / 2 + (rand() - 0.5) * 40;
    const y0 = Math.round(Math.min(Math.max(centrado, 4), ALTO - altoPx - 4));
    for (let y = 0; y < caja.alto; y++)
      for (let x = 0; x < caja.ancho; x++) {
        const v = img[(caja.y0 + y) * 28 + caja.x0 + x] / 255;
        if (v === 0) continue;
        for (let dy = 0; dy < s; dy++)
          for (let dx = 0; dx < s; dx++) {
            const k = (y0 + y * s + dy) * ANCHO + x0 + x * s + dx;
            if (v > lienzo[k]) lienzo[k] = v;
          }
      }
    // El trazo pasa a medir `multiplicador` veces el de MNIST: se engrosa lo que falta, proporcional al tamaño del dígito.
    const r = Math.round(((multiplicador - 1) * TRAZO_MNIST * s) / 2);
    engrosarRegion(lienzo, r, x0 - r, y0 - r, x0 + anchoPx + r, y0 + altoPx + r);
  });
  return lienzo;
}

describe.skipIf(!hayDatos)("secuencias sintéticas de MNIST (evaluación de la tubería completa)", () => {
  const modelo = cargarModelo(modeloJson as unknown as ModeloJSON);
  const { imagenes, etiquetas } = hayDatos ? cargarMnist() : { imagenes: [], etiquetas: [] };

  function evaluar(multiplicador: number) {
    const rand = aleatorio(SEMILLA + Math.round(multiplicador * 10));
    const porLargo = new Map<number, { secuencias: number; completas: number; segmentadas: number; digitos: number; digitosBien: number }>();
    let marcadasPegadas = 0;
    let conDudosos = 0;
    for (let k = 0; k < SECUENCIAS; k++) {
      const largo = 2 + (k % 5); // 2..6, repartido por igual
      const indices = Array.from({ length: largo }, () => Math.floor(rand() * imagenes.length));
      const numero = reconocerNumero(modelo, componer(imagenes, indices, rand, multiplicador), ANCHO, ALTO);
      const fila = porLargo.get(largo) ?? { secuencias: 0, completas: 0, segmentadas: 0, digitos: 0, digitosBien: 0 };
      fila.secuencias++;
      fila.digitos += largo;
      if (numero && numero.digitos.length === largo) {
        fila.segmentadas++;
        const bien = numero.digitos.filter((d, i) => !d.pegado && d.clasificacion.prediccion === etiquetas[indices[i]]).length;
        fila.digitosBien += bien;
        if (bien === largo) fila.completas++;
      }
      if (numero?.hayPegados) marcadasPegadas++;
      if (numero?.hayDudosos) conDudosos++;
      porLargo.set(largo, fila);
    }
    const total = [...porLargo.values()].reduce(
      (a, f) => ({
        secuencias: a.secuencias + f.secuencias,
        completas: a.completas + f.completas,
        segmentadas: a.segmentadas + f.segmentadas,
        digitos: a.digitos + f.digitos,
        digitosBien: a.digitosBien + f.digitosBien,
      }),
      { secuencias: 0, completas: 0, segmentadas: 0, digitos: 0, digitosBien: 0 },
    );
    return { porLargo, total, marcadasPegadas, conDudosos };
  }

  function informe(titulo: string, r: ReturnType<typeof evaluar>) {
    const pct = (a: number, b: number) => `${((100 * a) / b).toFixed(1)} %`;
    const lineas = [`\n${titulo}`, "largo | secuencias | bien segmentadas | dígitos bien | secuencia completa"];
    for (const [largo, f] of [...r.porLargo.entries()].sort((a, b) => a[0] - b[0])) {
      lineas.push(`  ${largo}   | ${f.secuencias}        | ${pct(f.segmentadas, f.secuencias)}           | ${pct(f.digitosBien, f.digitos)}       | ${pct(f.completas, f.secuencias)}`);
    }
    const t = r.total;
    lineas.push(
      `TOTAL | ${t.secuencias}        | ${pct(t.segmentadas, t.secuencias)}           | ${pct(t.digitosBien, t.digitos)}       | ${pct(t.completas, t.secuencias)}`,
      `avisos de «pegados»: ${r.marcadasPegadas}; secuencias con algún dígito dudoso: ${r.conDudosos}`,
    );
    console.log(lineas.join("\n"));
  }

  it("un dígito aislado por la tubería completa acierta casi como la red sola", () => {
    const rand = aleatorio(SEMILLA);
    let tuberia = 0;
    let sola = 0;
    const N = 2000;
    for (let k = 0; k < N; k++) {
      const indice = Math.floor(rand() * imagenes.length);
      const lienzo = componer(imagenes, [indice], rand, 1);
      const numero = reconocerNumero(modelo, lienzo, ANCHO, ALTO);
      if (numero?.digitos.length === 1 && numero.digitos[0].clasificacion.prediccion === etiquetas[indice]) tuberia++;
      const directa = clasificar(modelo, Float32Array.from(imagenes[indice], (p) => p / 255));
      if (directa.prediccion === etiquetas[indice]) sola++;
    }
    console.log(`
Un dígito aislado (${N} imágenes de prueba al azar): red sola sobre 28×28 = ${((100 * sola) / N).toFixed(2)} %; tubería completa sobre lienzo = ${((100 * tuberia) / N).toFixed(2)} %`);
    expect(tuberia / N).toBeGreaterThan(0.95);
  }, 120_000);

  it("trazo como el de MNIST: lee bien secuencias de 2 a 6 dígitos", () => {
    const r = evaluar(1);
    informe("Trazo como el de MNIST (dígitos ampliados de 3× a 9×)", r);
    expect(r.total.segmentadas / r.total.secuencias).toBeGreaterThan(0.95);
    expect(r.total.digitosBien / r.total.digitos).toBeGreaterThan(0.85);
  }, 120_000);

  it("trazo 1,3 veces más grueso (parecido al lápiz de la demo): sigue leyendo", () => {
    const r = evaluar(1.3);
    informe("Trazo 1,3× el de MNIST", r);
    expect(r.total.segmentadas / r.total.secuencias).toBeGreaterThan(0.95);
    expect(r.total.digitosBien / r.total.digitos).toBeGreaterThan(0.8);
  }, 120_000);

  it("trazo muy grueso (1,8×): prueba de estrés, solo informa", () => {
    const r = evaluar(1.8);
    informe("Trazo 1,8× el de MNIST (prueba de estrés)", r);
    expect(r.total.segmentadas / r.total.secuencias).toBeGreaterThan(0.9);
  }, 120_000);
});
