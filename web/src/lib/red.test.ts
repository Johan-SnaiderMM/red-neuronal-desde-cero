import { describe, expect, it } from "vitest";
import modeloJson from "../../public/modelo.json";
import ejemplos from "./__fixtures__/ejemplos.json";
import { cargarModelo, clasificar, logits, softmax, type ModeloJSON } from "./red";

const json = modeloJson as unknown as ModeloJSON;
const modelo = cargarModelo(json);

describe("red neuronal en TypeScript", () => {
  it("lee el modelo exportado desde Python", () => {
    expect(modelo.arquitectura).toEqual([784, 128, 64, 10]);
    expect(modelo.parametros).toBe(109_386);
    const total = modelo.capas.reduce((suma, c) => suma + c.W.length + c.b.length, 0);
    expect(total).toBe(modelo.parametros);
  });

  it("da los mismos logits que NumPy (paridad Python ↔ TypeScript)", () => {
    expect(ejemplos).toHaveLength(10);
    for (const e of ejemplos) {
      const entrada = Float32Array.from(e.pixeles, (p) => p / 255);
      const obtenidos = logits(modelo, entrada);
      e.logits.forEach((esperado, i) => expect(obtenidos[i]).toBeCloseTo(esperado, 3));
    }
  });

  it("elige la misma clase que NumPy y casi siempre acierta", () => {
    let aciertos = 0;
    for (const e of ejemplos) {
      const entrada = Float32Array.from(e.pixeles, (p) => p / 255);
      const { prediccion } = clasificar(modelo, entrada);
      const esperadaPorNumpy = e.logits.indexOf(Math.max(...e.logits));
      expect(prediccion).toBe(esperadaPorNumpy);
      if (prediccion === e.etiqueta) aciertos++;
    }
    expect(aciertos).toBeGreaterThanOrEqual(9);
  });

  it("softmax suma 1 y no se desborda con puntajes enormes", () => {
    const p = softmax(Float32Array.from([1000, 1001, 999]));
    expect(p.every(Number.isFinite)).toBe(true);
    expect(Array.from(p).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(p[1]).toBeGreaterThan(p[0]);
  });

  it("rechaza una entrada del tamaño equivocado", () => {
    expect(() => logits(modelo, new Float32Array(100))).toThrow(/784/);
  });

  it("detecta un modelo dañado o de otro formato", () => {
    const roto: ModeloJSON = { ...json, capas: [{ ...json.capas[0], W: "AAAA" }, ...json.capas.slice(1)] };
    expect(() => cargarModelo(roto)).toThrow(/dañado/);
    expect(() => cargarModelo({ ...json, formato: 2 })).toThrow(/no soportado/);
  });
});
