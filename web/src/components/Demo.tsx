"use client";

import clsx from "clsx";
import { useCallback, useEffect, useRef, useState } from "react";
import { LADO, prepararDigito } from "@/lib/preprocesar";
import { cargarModelo, clasificar, type Clasificacion, type Modelo, type ModeloJSON } from "@/lib/red";

/** Resolución interna del lienzo (se muestra más grande o más chico con CSS). */
const TAM = 280;
/** Grosor del trazo en píxeles del lienzo. */
const GROSOR = 22;
/** Por debajo de esta probabilidad la red avisa que no está segura. */
const CONFIANZA_MINIMA = 0.6;

function pintarFondo(lienzo: HTMLCanvasElement) {
  const ctx = lienzo.getContext("2d");
  if (!ctx) return;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, lienzo.width, lienzo.height);
}

export function Demo() {
  const lienzo = useRef<HTMLCanvasElement>(null);
  const vista = useRef<HTMLCanvasElement>(null);
  const dibujando = useRef(false);
  const ultimo = useRef<{ x: number; y: number } | null>(null);
  const programado = useRef(false);
  const [modelo, setModelo] = useState<Modelo | null>(null);
  const [errorModelo, setErrorModelo] = useState<string | null>(null);
  const [resultado, setResultado] = useState<Clasificacion | null>(null);

  // Descarga los pesos de la red (≈ 0,6 MB) una sola vez.
  useEffect(() => {
    let activo = true;
    fetch("/modelo.json")
      .then((r) => {
        if (!r.ok) throw new Error(`No pude descargar el modelo (${r.status}).`);
        return r.json() as Promise<ModeloJSON>;
      })
      .then((json) => activo && setModelo(cargarModelo(json)))
      .catch((e: unknown) => activo && setErrorModelo(e instanceof Error ? e.message : "No pude cargar el modelo."));
    return () => {
      activo = false;
    };
  }, []);

  useEffect(() => {
    if (lienzo.current) pintarFondo(lienzo.current);
    if (vista.current) pintarFondo(vista.current);
  }, []);

  const pintarVista = useCallback((entrada: Float32Array | null) => {
    const c = vista.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    const imagen = ctx.createImageData(LADO, LADO);
    for (let i = 0; i < LADO * LADO; i++) {
      const v = entrada ? Math.round(entrada[i] * 255) : 0;
      imagen.data[i * 4] = v;
      imagen.data[i * 4 + 1] = v;
      imagen.data[i * 4 + 2] = v;
      imagen.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imagen, 0, 0);
  }, []);

  const reconocer = useCallback(() => {
    const ctx = lienzo.current?.getContext("2d", { willReadFrequently: true });
    if (!ctx || !modelo) return;
    const pixeles = ctx.getImageData(0, 0, TAM, TAM).data;
    const tinta = new Float32Array(TAM * TAM);
    for (let i = 0; i < tinta.length; i++) tinta[i] = pixeles[i * 4] / 255;
    const entrada = prepararDigito(tinta, TAM, TAM);
    pintarVista(entrada);
    setResultado(entrada ? clasificar(modelo, entrada) : null);
  }, [modelo, pintarVista]);

  const posicion = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return { x: ((e.clientX - r.left) * TAM) / r.width, y: ((e.clientY - r.top) * TAM) / r.height };
  };

  const trazar = (desde: { x: number; y: number }, hasta: { x: number; y: number }) => {
    const ctx = lienzo.current?.getContext("2d");
    if (!ctx) return;
    ctx.strokeStyle = "#fff";
    ctx.fillStyle = "#fff";
    ctx.lineWidth = GROSOR;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(desde.x, desde.y);
    ctx.lineTo(hasta.x, hasta.y);
    ctx.stroke();
  };

  const alBajar = (e: React.PointerEvent<HTMLCanvasElement>) => {
    try {
      e.currentTarget.setPointerCapture(e.pointerId); // sigue el trazo aunque el dedo salga del lienzo
    } catch {
      // Algunos navegadores lanzan si el puntero ya no está activo: se dibuja igual mientras siga dentro.
    }
    dibujando.current = true;
    const p = posicion(e);
    ultimo.current = p;
    const ctx = lienzo.current?.getContext("2d");
    if (ctx) {
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(p.x, p.y, GROSOR / 2, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const alMover = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dibujando.current || !ultimo.current) return;
    const p = posicion(e);
    trazar(ultimo.current, p);
    ultimo.current = p;
    if (!programado.current) {
      programado.current = true;
      requestAnimationFrame(() => {
        programado.current = false;
        reconocer();
      });
    }
  };

  const alSoltar = () => {
    if (!dibujando.current) return;
    dibujando.current = false;
    ultimo.current = null;
    reconocer();
  };

  const borrar = () => {
    if (lienzo.current) pintarFondo(lienzo.current);
    pintarVista(null);
    setResultado(null);
  };

  const porcentaje = resultado ? Math.round(resultado.confianza * 100) : 0;
  const dudosa = resultado !== null && resultado.confianza < CONFIANZA_MINIMA;

  return (
    <section aria-labelledby="titulo-demo" className="rounded-3xl border border-linea bg-superficie p-4 sm:p-6">
      <h2 id="titulo-demo" className="sr-only">
        Demo: dibuja un dígito
      </h2>
      <div className="grid gap-6 md:grid-cols-2">
        <div>
          <p className="mb-2 text-sm text-tinta-2" id="ayuda-lienzo">
            Dibuja un dígito del 0 al 9 con el dedo o el mouse.
          </p>
          <canvas
            ref={lienzo}
            width={TAM}
            height={TAM}
            aria-label="Lienzo para dibujar un dígito"
            aria-describedby="ayuda-lienzo"
            className="aspect-square w-full max-w-sm cursor-crosshair touch-none rounded-2xl border-2 border-linea bg-black"
            onPointerDown={alBajar}
            onPointerMove={alMover}
            onPointerUp={alSoltar}
            onPointerCancel={alSoltar}
          />
          <button
            type="button"
            onClick={borrar}
            className="mt-3 min-h-12 rounded-xl border border-linea px-5 text-base font-medium text-tinta hover:bg-papel active:scale-[0.98]"
          >
            Borrar
          </button>
        </div>

        <div className="flex flex-col gap-4">
          <div aria-live="polite" className="min-h-20">
            {errorModelo ? (
              <p className="text-aviso">{errorModelo}</p>
            ) : !modelo ? (
              <p className="text-tinta-2">Cargando la red neuronal…</p>
            ) : resultado ? (
              <p>
                <span className="text-tinta-2">La red cree que es un </span>
                <strong className="align-middle text-5xl font-semibold text-acento">{resultado.prediccion}</strong>
                <span className="ml-2 text-tinta-2">({porcentaje} % de seguridad)</span>
                {dudosa && <span className="mt-1 block text-sm text-aviso">No estoy muy segura: prueba con un trazo más grande y claro.</span>}
              </p>
            ) : (
              <p className="text-tinta-2">Todavía no hay dibujo.</p>
            )}
          </div>

          <ul className="flex flex-col gap-1.5" aria-label="Probabilidad de cada dígito">
            {Array.from({ length: 10 }, (_, d) => {
              const p = resultado?.probabilidades[d] ?? 0;
              const gana = resultado?.prediccion === d;
              return (
                <li key={d} className="flex items-center gap-2">
                  <span className={clsx("w-4 text-center font-mono", gana ? "font-bold text-acento" : "text-tinta-2")}>{d}</span>
                  <div className="h-3 flex-1 overflow-hidden rounded-full bg-linea/60" aria-hidden="true">
                    <div className={clsx("h-full rounded-full transition-[width] duration-150", gana ? "bg-acento" : "bg-tinta-2/60")} style={{ width: `${(p * 100).toFixed(1)}%` }} />
                  </div>
                  <span className="w-[4.5rem] shrink-0 whitespace-nowrap text-right font-mono text-sm tabular-nums text-tinta-2">{(p * 100).toFixed(1)} %</span>
                </li>
              );
            })}
          </ul>

          <div className="flex items-center gap-3">
            <canvas
              ref={vista}
              width={LADO}
              height={LADO}
              aria-label="Lo que ve la red: tu dibujo reducido a 28 por 28 píxeles"
              className="size-28 rounded-lg border border-linea bg-black [image-rendering:pixelated]"
            />
            <p className="text-sm text-tinta-2">
              Lo que ve la red: tu dibujo reducido a 28×28 píxeles y centrado, igual que las imágenes con las que se entrenó.
            </p>
          </div>
        </div>
      </div>

      {modelo && (
        <p className="mt-5 border-t border-linea pt-3 text-sm text-tinta-2">
          Red {modelo.arquitectura.join("-")} · {modelo.parametros.toLocaleString("es-CO")} parámetros ·{" "}
          {(modelo.precisionPrueba * 100).toFixed(1).replace(".", ",")} % de aciertos en las 10 000 imágenes de prueba de MNIST.
        </p>
      )}
    </section>
  );
}
