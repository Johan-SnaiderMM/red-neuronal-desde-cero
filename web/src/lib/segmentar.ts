/**
 * Separa un dibujo con varios dígitos en un grupo de píxeles por dígito, de izquierda a derecha.
 *
 * La red solo sabe leer UN dígito a la vez (se entrenó con imágenes de MNIST, una cifra por imagen), así que para leer un
 * número hay que partir el dibujo antes de clasificar:
 *   1) se etiqueta cada trazo continuo (componente conexo, con 8 vecinos);
 *   2) los trazos de un mismo dígito (la raya de un 4, la barra de un 5) se reconocen porque se solapan en horizontal y se unen;
 *   3) se descartan las manchas y astillas pequeñas al lado de dígitos grandes;
 *   4) se ordenan por su borde izquierdo.
 * Límite conocido: dos dígitos que se TOCAN forman un solo trazo y quedan en un solo grupo.
 */

import { UMBRAL_TINTA, type Caja } from "./preprocesar";

export interface Componente {
  id: number;
  caja: Caja;
  /** Píxeles con tinta. */
  area: number;
}

export interface Grupo {
  caja: Caja;
  /** Etiquetas de los componentes que forman este grupo. */
  ids: number[];
  area: number;
}

export interface Segmentacion {
  /** Una etiqueta por píxel: 0 = fondo, 1..n = componente. */
  etiquetas: Int32Array;
  grupos: Grupo[];
}

/** Dos trazos son del mismo dígito si el solape horizontal supera esta fracción del más angosto. */
export const SOLAPE_MINIMO = 0.5;
/** Un grupo cuyo lado mayor mide menos de esta fracción del grupo más grande se considera una mancha. */
export const TAMANO_MINIMO_RELATIVO = 0.2;
/**
 * Un grupo con menos de esta fracción de la tinta del grupo más cargado es una astilla (por ejemplo, un trozo de un trazo fino
 * que se partió), no un dígito. Un «1» chico de trazo normal tiene mucha más tinta que eso.
 */
export const AREA_MINIMA_RELATIVA = 0.06;

export function etiquetarComponentes(
  tinta: Float32Array,
  ancho: number,
  alto: number,
): { etiquetas: Int32Array; componentes: Componente[] } {
  const etiquetas = new Int32Array(ancho * alto);
  const componentes: Componente[] = [];
  // Pila de píxeles por visitar. Cada píxel entra una sola vez (se etiqueta al entrar), así que nunca pasa de ancho×alto.
  const pila = new Int32Array(ancho * alto);

  for (let inicio = 0; inicio < etiquetas.length; inicio++) {
    if (etiquetas[inicio] !== 0 || tinta[inicio] <= UMBRAL_TINTA) continue;
    const id = componentes.length + 1;
    let tope = 0;
    pila[tope++] = inicio;
    etiquetas[inicio] = id;
    let minX = ancho,
      minY = alto,
      maxX = -1,
      maxY = -1,
      area = 0;
    while (tope > 0) {
      const p = pila[--tope];
      const y = Math.floor(p / ancho);
      const x = p - y * ancho;
      area++;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= alto) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= ancho) continue;
          const q = ny * ancho + nx;
          if (etiquetas[q] === 0 && tinta[q] > UMBRAL_TINTA) {
            etiquetas[q] = id;
            pila[tope++] = q;
          }
        }
      }
    }
    componentes.push({ id, caja: { izquierda: minX, arriba: minY, ancho: maxX - minX + 1, alto: maxY - minY + 1 }, area });
  }
  return { etiquetas, componentes };
}

/** Une componentes (Union-Find) cuyos rangos horizontales se solapan lo suficiente. */
function agruparPorSolape(componentes: Componente[]): Grupo[] {
  const padre = componentes.map((_, i) => i);
  const raiz = (i: number): number => {
    while (padre[i] !== i) {
      padre[i] = padre[padre[i]];
      i = padre[i];
    }
    return i;
  };
  for (let i = 0; i < componentes.length; i++) {
    for (let j = i + 1; j < componentes.length; j++) {
      const a = componentes[i].caja;
      const b = componentes[j].caja;
      const solape = Math.min(a.izquierda + a.ancho, b.izquierda + b.ancho) - Math.max(a.izquierda, b.izquierda);
      if (solape > 0 && solape >= SOLAPE_MINIMO * Math.min(a.ancho, b.ancho)) padre[raiz(i)] = raiz(j);
    }
  }

  const porRaiz = new Map<number, Grupo>();
  componentes.forEach((c, i) => {
    const r = raiz(i);
    const g = porRaiz.get(r);
    if (!g) {
      porRaiz.set(r, { caja: { ...c.caja }, ids: [c.id], area: c.area });
      return;
    }
    const derecha = Math.max(g.caja.izquierda + g.caja.ancho, c.caja.izquierda + c.caja.ancho);
    const abajo = Math.max(g.caja.arriba + g.caja.alto, c.caja.arriba + c.caja.alto);
    g.caja.izquierda = Math.min(g.caja.izquierda, c.caja.izquierda);
    g.caja.arriba = Math.min(g.caja.arriba, c.caja.arriba);
    g.caja.ancho = derecha - g.caja.izquierda;
    g.caja.alto = abajo - g.caja.arriba;
    g.ids.push(c.id);
    g.area += c.area;
  });
  return [...porRaiz.values()];
}

export function segmentar(tinta: Float32Array, ancho: number, alto: number): Segmentacion {
  const { etiquetas, componentes } = etiquetarComponentes(tinta, ancho, alto);
  let grupos = agruparPorSolape(componentes);

  // Si hay varios grupos, se tiran los que son una mancha frente al más grande (un toque accidental con el dedo).
  // Si solo hay uno, se conserva aunque sea pequeño: no es decisión del programa borrar lo único que se dibujó.
  if (grupos.length > 1) {
    const mayor = Math.max(...grupos.map((g) => Math.max(g.caja.ancho, g.caja.alto)));
    const masTinta = Math.max(...grupos.map((g) => g.area));
    grupos = grupos.filter(
      (g) => Math.max(g.caja.ancho, g.caja.alto) >= TAMANO_MINIMO_RELATIVO * mayor && g.area >= AREA_MINIMA_RELATIVA * masTinta,
    );
  }
  grupos.sort((a, b) => a.caja.izquierda - b.caja.izquierda);
  return { etiquetas, grupos };
}

/**
 * Recorta la caja del grupo y deja solo su tinta: lo que sea de OTRO grupo dentro de esa caja se pone en 0.
 * Así un vecino que se asoma por la esquina no se cuela en la imagen del dígito.
 */
export function recortarGrupo(
  tinta: Float32Array,
  etiquetas: Int32Array,
  anchoLienzo: number,
  grupo: Grupo,
): { datos: Float32Array; ancho: number; alto: number } {
  const { izquierda, arriba, ancho, alto } = grupo.caja;
  const propios = new Set(grupo.ids);
  const datos = new Float32Array(ancho * alto);
  for (let y = 0; y < alto; y++) {
    for (let x = 0; x < ancho; x++) {
      const i = (arriba + y) * anchoLienzo + izquierda + x;
      const etiqueta = etiquetas[i];
      if (etiqueta === 0 || propios.has(etiqueta)) datos[y * ancho + x] = tinta[i];
    }
  }
  return { datos, ancho, alto };
}
