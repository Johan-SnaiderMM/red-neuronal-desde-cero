import { describe, expect, it } from "vitest";
import modeloJson from "../../public/modelo.json";
import ejemplos from "./__fixtures__/ejemplos.json";
import { CAJA, LADO, cajaDelTrazo, centroDeMasa, prepararDigito, remuestrearPorArea } from "./preprocesar";
import { cargarModelo, clasificar, type ModeloJSON } from "./red";

const modelo = cargarModelo(modeloJson as unknown as ModeloJSON);

function lienzo(lado: number) {
  return new Float32Array(lado * lado);
}

function rectangulo(l: Float32Array, lado: number, x0: number, y0: number, x1: number, y1: number) {
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) l[y * lado + x] = 1;
}

describe("prepararDigito", () => {
  it("devuelve null si no hay trazo", () => {
    expect(prepararDigito(lienzo(280), 280, 280)).toBeNull();
    expect(cajaDelTrazo(lienzo(50), 50, 50)).toBeNull();
  });

  it("encuadra el trazo en 20 px y lo centra por su centro de masa", () => {
    const l = lienzo(280);
    rectangulo(l, 280, 10, 10, 40, 200); // barra vertical pegada a la esquina
    const salida = prepararDigito(l, 280, 280)!;
    expect(salida).toHaveLength(LADO * LADO);
    const caja = cajaDelTrazo(salida, LADO, LADO)!;
    expect(Math.max(caja.ancho, caja.alto)).toBeLessThanOrEqual(CAJA);
    const centro = centroDeMasa(salida, LADO, LADO)!;
    expect(Math.abs(centro.x - 13.5)).toBeLessThan(0.7);
    expect(Math.abs(centro.y - 13.5)).toBeLessThan(0.7);
    expect(Math.max(...salida)).toBeLessThanOrEqual(1);
  });

  it("da el mismo resultado sin importar dónde ni de qué tamaño se dibuje", () => {
    const chico = lienzo(280);
    rectangulo(chico, 280, 200, 30, 220, 90);
    const grande = lienzo(280);
    rectangulo(grande, 280, 20, 20, 80, 200); // 60×180: misma proporción 1:3 (20×60), el triple de grande
    const a = prepararDigito(chico, 280, 280)!;
    const b = prepararDigito(grande, 280, 280)!;
    let diferencia = 0;
    for (let i = 0; i < a.length; i++) diferencia += Math.abs(a[i] - b[i]);
    expect(diferencia / a.length).toBeLessThan(0.02);
  });

  it("el promedio por área conserva la intensidad media", () => {
    const origen = Float32Array.from({ length: 16 }, (_, i) => (i % 2 === 0 ? 1 : 0));
    const reducida = remuestrearPorArea(origen, 4, { izquierda: 0, arriba: 0, ancho: 4, alto: 4 }, 2, 2);
    expect(Array.from(reducida)).toEqual([0.5, 0.5, 0.5, 0.5]);
  });

  it("reconoce dígitos de prueba ampliados y puestos en otro lugar del lienzo, como si se hubieran dibujado", () => {
    const LADO_LIENZO = 320;
    let aciertos = 0;
    for (const e of ejemplos) {
      const l = lienzo(LADO_LIENZO);
      // Ampliación ×10 con vecino más cercano, desplazada para no quedar centrada.
      for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
          const v = e.pixeles[y * 28 + x] / 255;
          if (v > 0) for (let k = 0; k < 100; k++) l[(17 + y * 10 + Math.floor(k / 10)) * LADO_LIENZO + 23 + x * 10 + (k % 10)] = v;
        }
      }
      const entrada = prepararDigito(l, LADO_LIENZO, LADO_LIENZO)!;
      if (clasificar(modelo, entrada).prediccion === e.etiqueta) aciertos++;
    }
    expect(aciertos).toBeGreaterThanOrEqual(8);
  });
});
