import { describe, expect, it } from "vitest";
import { etiquetarComponentes, recortarGrupo, segmentar } from "./segmentar";

const ANCHO = 840;
const ALTO = 280;

function lienzo() {
  return new Float32Array(ANCHO * ALTO);
}

/** Rectángulo de tinta: x0..x1 y y0..y1 son exclusivos al final. */
function rect(l: Float32Array, x0: number, y0: number, x1: number, y1: number, valor = 1) {
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) l[y * ANCHO + x] = valor;
}

describe("etiquetarComponentes", () => {
  it("no encuentra nada en un lienzo vacío", () => {
    expect(etiquetarComponentes(lienzo(), ANCHO, ALTO).componentes).toHaveLength(0);
  });

  it("cuenta el área y la caja de cada trazo separado", () => {
    const l = lienzo();
    rect(l, 10, 20, 30, 120); // 20×100
    rect(l, 100, 40, 110, 50); // 10×10
    const { componentes } = etiquetarComponentes(l, ANCHO, ALTO);
    expect(componentes).toHaveLength(2);
    expect(componentes[0]).toMatchObject({ area: 2000, caja: { izquierda: 10, arriba: 20, ancho: 20, alto: 100 } });
    expect(componentes[1]).toMatchObject({ area: 100, caja: { izquierda: 100, arriba: 40, ancho: 10, alto: 10 } });
  });

  it("une píxeles que solo se tocan en diagonal (8 vecinos)", () => {
    const l = lienzo();
    l[10 * ANCHO + 10] = 1;
    l[11 * ANCHO + 11] = 1;
    expect(etiquetarComponentes(l, ANCHO, ALTO).componentes).toHaveLength(1);
  });

  it("ignora la tinta por debajo del umbral", () => {
    const l = lienzo();
    rect(l, 10, 10, 20, 20, 0.05);
    expect(etiquetarComponentes(l, ANCHO, ALTO).componentes).toHaveLength(0);
  });
});

describe("segmentar", () => {
  it("devuelve cero grupos si no hay trazo", () => {
    expect(segmentar(lienzo(), ANCHO, ALTO).grupos).toEqual([]);
  });

  it("un solo dígito es un solo grupo", () => {
    const l = lienzo();
    rect(l, 50, 40, 90, 200);
    const { grupos } = segmentar(l, ANCHO, ALTO);
    expect(grupos).toHaveLength(1);
    expect(grupos[0].caja).toEqual({ izquierda: 50, arriba: 40, ancho: 40, alto: 160 });
  });

  it("ordena los dígitos de izquierda a derecha sin importar en qué orden se dibujaron", () => {
    const l = lienzo();
    rect(l, 600, 40, 640, 200); // se dibuja primero el de la derecha
    rect(l, 300, 50, 340, 210);
    rect(l, 40, 30, 80, 190);
    const { grupos } = segmentar(l, ANCHO, ALTO);
    expect(grupos.map((g) => g.caja.izquierda)).toEqual([40, 300, 600]);
  });

  it("trata como un solo dígito los trazos que se solapan en horizontal (un 4 hecho con dos trazos)", () => {
    const l = lienzo();
    rect(l, 60, 40, 80, 142); // trazo 1: palo izquierdo...
    rect(l, 60, 120, 160, 142); // ...y la raya horizontal (una «L» ancha)
    rect(l, 100, 40, 122, 110); // trazo 2: palo derecho, separado de la raya por un hueco de 10 px
    expect(etiquetarComponentes(l, ANCHO, ALTO).componentes).toHaveLength(2);
    const { grupos } = segmentar(l, ANCHO, ALTO);
    expect(grupos).toHaveLength(1);
    expect(grupos[0].caja).toEqual({ izquierda: 60, arriba: 40, ancho: 100, alto: 102 });
  });

  it("separa dos dígitos vecinos aunque haya solo 3 px de hueco entre ellos", () => {
    const l = lienzo();
    rect(l, 40, 40, 100, 200);
    rect(l, 103, 40, 163, 200);
    expect(segmentar(l, ANCHO, ALTO).grupos).toHaveLength(2);
  });

  it("deja como un solo grupo los dígitos que se tocan (límite conocido)", () => {
    const l = lienzo();
    rect(l, 40, 40, 100, 200);
    rect(l, 100, 40, 160, 200); // pegados sin hueco
    expect(segmentar(l, ANCHO, ALTO).grupos).toHaveLength(1);
  });

  it("descarta una mancha pequeña al lado de dígitos grandes", () => {
    const l = lienzo();
    rect(l, 40, 40, 90, 200);
    rect(l, 300, 40, 350, 200);
    rect(l, 200, 100, 215, 115); // toque accidental de 15×15
    expect(segmentar(l, ANCHO, ALTO).grupos).toHaveLength(2);
  });

  it("descarta una astilla fina (un trozo de trazo partido) que no es un dígito", () => {
    const l = lienzo();
    rect(l, 40, 40, 100, 200); // 60×160
    rect(l, 300, 40, 360, 200);
    rect(l, 150, 60, 155, 160); // astilla de 5×100: alta, pero con ~5 % de la tinta de un dígito
    expect(segmentar(l, ANCHO, ALTO).grupos).toHaveLength(2);
  });

  it("conserva un «1» delgado y chico entre dígitos grandes", () => {
    const l = lienzo();
    rect(l, 40, 20, 140, 260); // un dígito grande y gordo
    rect(l, 300, 120, 322, 200); // un «1» de trazo de 22 px y 80 de alto
    rect(l, 500, 20, 600, 260);
    expect(segmentar(l, ANCHO, ALTO).grupos).toHaveLength(3);
  });

  it("si solo hay una mancha pequeña, la conserva (no decide por el usuario)", () => {
    const l = lienzo();
    rect(l, 200, 100, 215, 115);
    expect(segmentar(l, ANCHO, ALTO).grupos).toHaveLength(1);
  });

  it("no mezcla trazos de dígitos distintos aunque sean de tamaños muy diferentes", () => {
    const l = lienzo();
    rect(l, 40, 20, 80, 260); // alto
    rect(l, 200, 120, 240, 180); // bajo (25 % del alto del otro: se conserva)
    expect(segmentar(l, ANCHO, ALTO).grupos).toHaveLength(2);
  });
});

describe("recortarGrupo", () => {
  it("copia solo los píxeles del grupo: un vecino dentro de su caja no lo contamina", () => {
    const l = lienzo();
    rect(l, 40, 40, 120, 60); // A: barra de arriba, 80 de ancho
    rect(l, 40, 40, 60, 200); // A: palo izquierdo
    rect(l, 110, 100, 170, 200); // B: solapa 10 px de ancho con A (menos de la mitad de su ancho)
    const seg = segmentar(l, ANCHO, ALTO);
    expect(seg.grupos).toHaveLength(2);
    const a = recortarGrupo(l, seg.etiquetas, ANCHO, seg.grupos[0]);
    expect(a.ancho).toBe(80);
    expect(a.alto).toBe(160);
    // En la esquina inferior derecha de la caja de A (x 110..119, y 100..199) está el vecino B: en el recorte debe ser 0.
    const xRel = 112 - 40;
    const yRel = 150 - 40;
    expect(a.datos[yRel * a.ancho + xRel]).toBe(0);
    // Y su propia tinta sí está.
    expect(a.datos[10 * a.ancho + 10]).toBe(1);
  });
});
