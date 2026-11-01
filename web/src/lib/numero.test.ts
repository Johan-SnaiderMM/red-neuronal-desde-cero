import { describe, expect, it } from "vitest";
import modeloJson from "../../public/modelo.json";
import ejemplos from "./__fixtures__/ejemplos.json";
import { CONFIANZA_MINIMA, reconocerNumero } from "./numero";
import { prepararDigito } from "./preprocesar";
import { cargarModelo, clasificar, type ModeloJSON } from "./red";

const modelo = cargarModelo(modeloJson as unknown as ModeloJSON);
const ANCHO = 840;
const ALTO = 280;
const ESCALA = 8; // cada píxel de MNIST pasa a ser un cuadrado de 8×8 en el lienzo (un dígito de 28×28 mide 224 px)

/**
 * Pega los ejemplos (imágenes de MNIST 28×28) uno al lado del otro en un lienzo ancho, ampliados con vecino más cercano,
 * como si se hubieran dibujado. `paso` es la distancia entre los bordes izquierdos de dos cuadros consecutivos.
 */
function componer(indices: number[], paso = 28 * ESCALA, escala = ESCALA, inicioX = 20, inicioY = 20): Float32Array {
  const l = new Float32Array(ANCHO * ALTO);
  indices.forEach((indice, k) => {
    const e = ejemplos[indice];
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const v = e.pixeles[y * 28 + x] / 255;
        if (v === 0) continue;
        for (let dy = 0; dy < escala; dy++) {
          for (let dx = 0; dx < escala; dx++) {
            const px = inicioX + k * paso + x * escala + dx;
            const py = inicioY + y * escala + dy;
            if (px < ANCHO && py < ALTO) l[py * ANCHO + px] = Math.max(l[py * ANCHO + px], v);
          }
        }
      }
    }
  });
  return l;
}

describe("reconocerNumero", () => {
  it("devuelve null si no hay trazo", () => {
    expect(reconocerNumero(modelo, new Float32Array(ANCHO * ALTO), ANCHO, ALTO)).toBeNull();
  });

  it("con un solo dígito da lo mismo que preparar y clasificar todo el lienzo (sin regresión)", () => {
    for (let i = 0; i < ejemplos.length; i++) {
      const l = componer([i]);
      const numero = reconocerNumero(modelo, l, ANCHO, ALTO)!;
      expect(numero.digitos).toHaveLength(1);
      const entradaAntigua = prepararDigito(l, ANCHO, ALTO)!;
      const antigua = clasificar(modelo, entradaAntigua);
      expect(numero.digitos[0].clasificacion.prediccion).toBe(antigua.prediccion);
      expect(numero.digitos[0].clasificacion.confianza).toBeCloseTo(antigua.confianza, 2);
    }
  });

  it("separa y lee varios dígitos en orden", () => {
    const orden = [3, 0, 7, 1, 9, 4];
    for (let n = 2; n <= orden.length; n++) {
      const indices = orden.slice(0, n);
      // A escala 4 caben 6 dígitos de 112 px en los 840 px del lienzo.
      const numero = reconocerNumero(modelo, componer(indices, 28 * 4, 4), ANCHO, ALTO)!;
      expect(numero.digitos).toHaveLength(n);
      expect(numero.hayPegados).toBe(false);
      // Orden de izquierda a derecha.
      const lefts = numero.digitos.map((d) => d.caja.izquierda);
      expect([...lefts].sort((a, b) => a - b)).toEqual(lefts);
    }
  });

  it("lee casi todos los dígitos de prueba puestos en secuencias de 3", () => {
    let total = 0;
    let aciertos = 0;
    for (let inicio = 0; inicio + 3 <= ejemplos.length; inicio++) {
      const indices = [inicio, inicio + 1, inicio + 2];
      const numero = reconocerNumero(modelo, componer(indices), ANCHO, ALTO)!;
      expect(numero.digitos).toHaveLength(3);
      numero.digitos.forEach((d, k) => {
        total++;
        if (d.clasificacion.prediccion === ejemplos[indices[k]].etiqueta) aciertos++;
      });
    }
    expect(aciertos / total).toBeGreaterThanOrEqual(0.85);
  });

  it("avisa cuando dos dígitos están pegados y no inventa un resultado", () => {
    // Paso de 20 cuadros en vez de 28: los dígitos (que ocupan unos 20 cuadros) se tocan.
    const numero = reconocerNumero(modelo, componer([3, 5], 20 * ESCALA), ANCHO, ALTO)!;
    expect(numero.hayPegados).toBe(true);
    expect(numero.texto).toContain("?");
  });

  it("marca como dudoso un dígito cuya seguridad es menor que el umbral", () => {
    const numero = reconocerNumero(modelo, componer([0, 1, 2]), ANCHO, ALTO)!;
    for (const d of numero.digitos) expect(d.dudoso).toBe(d.clasificacion.confianza < CONFIANZA_MINIMA);
    expect(numero.hayDudosos).toBe(numero.digitos.some((d) => d.dudoso));
  });
});
