"use client";

import clsx from "clsx";
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { LADO } from "@/lib/preprocesar";
import { CONFIANZA_MINIMA, reconocerNumero, type NumeroReconocido } from "@/lib/numero";
import { cargarModelo, type Modelo, type ModeloJSON } from "@/lib/red";

/** Alto interno del lienzo (se muestra más grande o más chico con CSS). */
const ALTO = 280;
/** Ancho interno: 3:1 en pantallas grandes (caben unos 6 dígitos) y 2:1 en celulares (unos 4). */
const ANCHO_GRANDE = 840;
const ANCHO_CELULAR = 560;
/** Grosor del trazo en píxeles del lienzo. */
const GROSOR = 22;
/** Espera tras soltar el dedo antes de leer: deja terminar dígitos de varios trazos (el 4, el 5) y empezar el siguiente. */
const PAUSA_MS = 450;

const CONSULTA_CELULAR = "(max-width: 639px)";

/** `true` en pantallas angostas. En el servidor (y en la primera pintura) se asume pantalla grande. */
function useEsCelular(): boolean {
  return useSyncExternalStore(
    (avisar) => {
      const mq = window.matchMedia(CONSULTA_CELULAR);
      mq.addEventListener("change", avisar);
      return () => mq.removeEventListener("change", avisar);
    },
    () => window.matchMedia(CONSULTA_CELULAR).matches,
    () => false,
  );
}

function pintarFondo(lienzo: HTMLCanvasElement) {
  const ctx = lienzo.getContext("2d");
  if (!ctx) return;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, lienzo.width, lienzo.height);
}

/** Dibuja los 28×28 que ve la red para un dígito. */
function Miniatura({ entrada }: { entrada: Float32Array }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    const imagen = ctx.createImageData(LADO, LADO);
    for (let i = 0; i < LADO * LADO; i++) {
      const v = Math.round(entrada[i] * 255);
      imagen.data[i * 4] = v;
      imagen.data[i * 4 + 1] = v;
      imagen.data[i * 4 + 2] = v;
      imagen.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imagen, 0, 0);
  }, [entrada]);
  return <canvas ref={ref} width={LADO} height={LADO} aria-hidden="true" className="size-14 rounded-md border border-linea bg-black [image-rendering:pixelated]" />;
}

export function Demo() {
  const lienzo = useRef<HTMLCanvasElement>(null);
  const dibujando = useRef(false);
  const ultimo = useRef<{ x: number; y: number } | null>(null);
  const temporizador = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ancho = useEsCelular() ? ANCHO_CELULAR : ANCHO_GRANDE;
  const [modelo, setModelo] = useState<Modelo | null>(null);
  const [errorModelo, setErrorModelo] = useState<string | null>(null);
  // Se guarda con el ancho del lienzo en que se leyó: si la pantalla gira, el lienzo se vacía y la lectura vieja ya no vale.
  const [lectura, setLectura] = useState<{ numero: NumeroReconocido; ancho: number } | null>(null);
  const numero = lectura && lectura.ancho === ancho ? lectura.numero : null;
  const [elegido, setElegido] = useState(0);

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

  // Cambiar el ancho vacía el lienzo: se vuelve a pintar el fondo.
  useEffect(() => {
    if (lienzo.current) pintarFondo(lienzo.current);
  }, [ancho]);

  useEffect(
    () => () => {
      if (temporizador.current) clearTimeout(temporizador.current);
    },
    [],
  );

  const reconocer = useCallback(() => {
    const ctx = lienzo.current?.getContext("2d", { willReadFrequently: true });
    if (!ctx || !modelo) return;
    const pixeles = ctx.getImageData(0, 0, ancho, ALTO).data;
    const tinta = new Float32Array(ancho * ALTO);
    for (let i = 0; i < tinta.length; i++) tinta[i] = pixeles[i * 4] / 255;
    const leido = reconocerNumero(modelo, tinta, ancho, ALTO);
    setLectura(leido ? { numero: leido, ancho } : null);
    setElegido((previo) => (leido ? Math.min(previo, leido.digitos.length - 1) : 0));
  }, [modelo, ancho]);

  const programarLectura = useCallback(() => {
    if (temporizador.current) clearTimeout(temporizador.current);
    temporizador.current = setTimeout(reconocer, PAUSA_MS);
  }, [reconocer]);

  const posicion = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return { x: ((e.clientX - r.left) * ancho) / r.width, y: ((e.clientY - r.top) * ALTO) / r.height };
  };

  const trazar = (desde: { x: number; y: number }, hasta: { x: number; y: number }) => {
    const ctx = lienzo.current?.getContext("2d");
    if (!ctx) return;
    ctx.strokeStyle = "#fff";
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
    if (temporizador.current) clearTimeout(temporizador.current); // sigue escribiendo: aún no hay que leer
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
  };

  const alSoltar = () => {
    if (!dibujando.current) return;
    dibujando.current = false;
    ultimo.current = null;
    programarLectura();
  };

  const borrar = () => {
    if (temporizador.current) clearTimeout(temporizador.current);
    if (lienzo.current) pintarFondo(lienzo.current);
    setLectura(null);
    setElegido(0);
  };

  const digitoElegido = numero?.digitos[Math.min(elegido, (numero?.digitos.length ?? 1) - 1)] ?? null;
  const todosPegados = numero?.digitos.every((d) => d.pegado) ?? false;
  const dudosos = numero?.digitos.map((d, i) => (d.dudoso && !d.pegado ? i + 1 : 0)).filter(Boolean) ?? [];

  return (
    <section aria-labelledby="titulo-demo" className="rounded-3xl border border-linea bg-superficie p-4 sm:p-6">
      <h2 id="titulo-demo" className="sr-only">
        Demo: escribe un número
      </h2>

      <p className="mb-2 text-sm text-tinta-2" id="ayuda-lienzo">
        Escribe un número con el dedo o el mouse: un dígito o varios, separados entre sí (hasta unos 6; en celular caben unos 4). La red lo lee cuando
        haces una pausa.
      </p>
      <div className="relative">
        <canvas
          ref={lienzo}
          width={ancho}
          height={ALTO}
          aria-label="Lienzo para escribir un número"
          aria-describedby="ayuda-lienzo"
          style={{ aspectRatio: `${ancho} / ${ALTO}` }}
          className="w-full cursor-crosshair touch-none rounded-2xl border-2 border-linea bg-black"
          onPointerDown={alBajar}
          onPointerMove={alMover}
          onPointerUp={alSoltar}
          onPointerCancel={alSoltar}
        />
        {numero && (
          <svg viewBox={`0 0 ${ancho} ${ALTO}`} className="pointer-events-none absolute inset-0 size-full" aria-hidden="true">
            {numero.digitos.map((d, i) => (
              <g key={i}>
                <rect
                  x={d.caja.izquierda - 6}
                  y={d.caja.arriba - 6}
                  width={d.caja.ancho + 12}
                  height={d.caja.alto + 12}
                  rx={10}
                  fill="none"
                  strokeWidth={i === elegido ? 5 : 3}
                  className={d.dudoso ? "stroke-aviso" : "stroke-acento"}
                />
                <text x={d.caja.izquierda - 2} y={Math.max(d.caja.arriba - 12, 22)} fontSize={22} fontWeight={700} className={d.dudoso ? "fill-aviso" : "fill-acento"}>
                  {i + 1}
                </text>
              </g>
            ))}
          </svg>
        )}
      </div>
      <button
        type="button"
        onClick={borrar}
        className="mt-3 min-h-12 rounded-xl border border-linea px-5 text-base font-medium text-tinta hover:bg-papel active:scale-[0.98]"
      >
        Borrar
      </button>

      <div className="mt-6 grid gap-6 md:grid-cols-2">
        <div aria-live="polite" className="min-h-24">
          {errorModelo ? (
            <p className="text-aviso">{errorModelo}</p>
          ) : !modelo ? (
            <p className="text-tinta-2">Cargando la red neuronal…</p>
          ) : numero ? (
            <div className="flex flex-col gap-2">
              {!todosPegados && (
                <p>
                  <span className="text-tinta-2">{numero.digitos.length === 1 ? "La red cree que es un " : "La red lee el número "}</span>
                  <strong className="align-middle font-mono text-5xl font-semibold tracking-wider text-acento">{numero.texto}</strong>
                </p>
              )}
              {numero.hayPegados && (
                <p className="text-sm text-aviso">Parece que hay dígitos pegados. Sepáralos un poco: la red solo lee un dígito a la vez.</p>
              )}
              {dudosos.length > 0 && (
                <p className="text-sm text-aviso">
                  No estoy muy segura {dudosos.length === 1 ? `del dígito ${dudosos[0]}` : `de los dígitos ${dudosos.join(", ")}`}: prueba con un trazo más
                  grande y claro.
                </p>
              )}
            </div>
          ) : (
            <p className="text-tinta-2">Todavía no hay dibujo.</p>
          )}
        </div>

        {numero && (
          <ul className="flex flex-wrap gap-2" aria-label="Dígitos detectados">
            {numero.digitos.map((d, i) => (
              <li key={i}>
                <button
                  type="button"
                  onClick={() => setElegido(i)}
                  aria-pressed={i === elegido}
                  className={clsx(
                    "flex min-h-12 items-center gap-2 rounded-xl border px-2 py-1.5 text-left hover:bg-papel",
                    i === elegido ? "border-acento bg-papel" : "border-linea",
                  )}
                >
                  <Miniatura entrada={d.entrada} />
                  <span className="flex flex-col leading-tight">
                    <span className="text-xs text-tinta-2">Dígito {i + 1}</span>
                    <span className={clsx("font-mono text-xl font-semibold", d.dudoso ? "text-aviso" : "text-acento")}>
                      {d.pegado ? "?" : d.clasificacion.prediccion}
                    </span>
                    <span className="font-mono text-xs tabular-nums text-tinta-2">
                      {d.pegado ? "no fiable" : `${(d.clasificacion.confianza * 100).toFixed(0)} %`}
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {digitoElegido?.pegado && (
        <p className="mt-6 text-sm text-tinta-2">Este recuadro parece tener más de un dígito, así que no se muestran probabilidades: la red solo sabe leer uno.</p>
      )}

      {digitoElegido && !digitoElegido.pegado && numero && (
        <div className="mt-6">
          <h3 className="mb-2 text-sm font-medium text-tinta">
            Probabilidad de cada dígito para el dígito {Math.min(elegido, numero.digitos.length - 1) + 1}
            {numero.digitos.length > 1 ? " (toca otra ficha para cambiar)" : ""}
          </h3>
          <ul className="flex flex-col gap-1.5" aria-label="Probabilidad de cada dígito">
            {Array.from({ length: 10 }, (_, d) => {
              const p = digitoElegido.clasificacion.probabilidades[d] ?? 0;
              const gana = digitoElegido.clasificacion.prediccion === d;
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
        </div>
      )}

      {numero && (
        <p className="mt-4 text-sm text-tinta-2">
          Cada recuadro del lienzo es un dígito que encontró la red: se recorta, se reduce a 28×28 píxeles y se centra, igual que las imágenes con las que se
          entrenó (las miniaturas de arriba). El umbral de duda es {Math.round(CONFIANZA_MINIMA * 100)} % de seguridad.
        </p>
      )}

      {modelo && (
        <p className="mt-5 border-t border-linea pt-3 text-sm text-tinta-2">
          Red {modelo.arquitectura.join("-")} · {modelo.parametros.toLocaleString("es-CO")} parámetros ·{" "}
          {(modelo.precisionPrueba * 100).toFixed(1).replace(".", ",")} % de aciertos en las 10 000 imágenes de prueba de MNIST (un dígito aislado).
        </p>
      )}
    </section>
  );
}
