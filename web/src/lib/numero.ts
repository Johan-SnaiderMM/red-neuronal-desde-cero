/**
 * Lee un número escrito a mano: parte el dibujo en dígitos (`segmentar`), prepara cada uno como una imagen de MNIST
 * (`prepararDigito`) y los clasifica con la red, que sigue siendo la misma de un solo dígito.
 */

import { prepararDigito, type Caja } from "./preprocesar";
import { clasificar, type Clasificacion, type Modelo } from "./red";
import { recortarGrupo, segmentar } from "./segmentar";

/** Por debajo de esta probabilidad la red avisa que no está segura. */
export const CONFIANZA_MINIMA = 0.6;
/**
 * Un dígito suelto es más alto que ancho o casi cuadrado. Si la caja de un grupo es bastante más ancha que alta, lo más
 * probable es que sean dos dígitos pegados.
 */
export const PROPORCION_PEGADOS = 1.25;

export interface DigitoReconocido {
  /** Caja del dígito en el lienzo (para dibujarla encima). */
  caja: Caja;
  /** Los 784 valores que ve la red. */
  entrada: Float32Array;
  clasificacion: Clasificacion;
  /** Seguridad menor que `CONFIANZA_MINIMA`. */
  dudoso: boolean;
  /** Parece haber más de un dígito pegado dentro de esta caja: la lectura no es fiable. */
  pegado: boolean;
}

export interface NumeroReconocido {
  digitos: DigitoReconocido[];
  /** Dígitos leídos de izquierda a derecha; «?» donde hay dígitos pegados. */
  texto: string;
  hayDudosos: boolean;
  hayPegados: boolean;
}

/** `tinta`: intensidades 0-1 del lienzo (1 = trazo). Devuelve null si no hay nada dibujado. */
export function reconocerNumero(modelo: Modelo, tinta: Float32Array, ancho: number, alto: number): NumeroReconocido | null {
  const { etiquetas, grupos } = segmentar(tinta, ancho, alto);
  const digitos: DigitoReconocido[] = [];
  for (const grupo of grupos) {
    const recorte = recortarGrupo(tinta, etiquetas, ancho, grupo);
    const entrada = prepararDigito(recorte.datos, recorte.ancho, recorte.alto);
    if (!entrada) continue;
    const clasificacion = clasificar(modelo, entrada);
    const pegado = grupo.caja.ancho / grupo.caja.alto > PROPORCION_PEGADOS;
    digitos.push({
      caja: grupo.caja,
      entrada,
      clasificacion,
      dudoso: clasificacion.confianza < CONFIANZA_MINIMA,
      pegado,
    });
  }
  if (digitos.length === 0) return null;
  return {
    digitos,
    texto: digitos.map((d) => (d.pegado ? "?" : String(d.clasificacion.prediccion))).join(""),
    hayDudosos: digitos.some((d) => d.dudoso),
    hayPegados: digitos.some((d) => d.pegado),
  };
}
