import { Demo } from "@/components/Demo";

const REPOSITORIO = "https://github.com/Johan-SnaiderMM/red-neuronal-desde-cero";

export default function Inicio() {
  return (
    <main className="mx-auto flex w-full max-w-4xl flex-col gap-8 px-4 py-8 sm:px-6 sm:py-12">
      <header className="flex flex-col gap-3">
        <p className="font-mono text-sm text-acento">NumPy · retropropagación a mano · corre en tu navegador</p>
        <h1 className="text-balance text-3xl font-semibold tracking-tight sm:text-5xl">Red neuronal desde cero</h1>
        <p className="max-w-2xl text-pretty text-lg text-tinta-2">
          Escribe un número, de uno o de varios dígitos. Una red neuronal que escribí con NumPy, sin PyTorch ni TensorFlow y con la
          retropropagación hecha a mano, lo lee aquí mismo. Tu dibujo no sale de tu dispositivo.
        </p>
      </header>

      <Demo />

      <section className="grid gap-6 md:grid-cols-2" aria-labelledby="como-funciona">
        <div className="flex flex-col gap-3">
          <h2 id="como-funciona" className="text-xl font-semibold">
            Cómo funciona
          </h2>
          <ol className="flex list-decimal flex-col gap-2 pl-5 text-tinta-2">
            <li>
              Se separa el dibujo en dígitos: cada trazo continuo es un candidato y los trazos que se solapan en horizontal (la raya de un 4) se
              unen. Se leen de izquierda a derecha.
            </li>
            <li>Cada dígito se reduce a 28×28 píxeles y se centra, como las imágenes de MNIST: son 784 números entre 0 y 1.</li>
            <li>Pasa por dos capas ocultas de 128 y 64 neuronas con activación ReLU.</li>
            <li>La última capa entrega 10 puntajes; softmax los convierte en probabilidades, una por dígito. La red es la misma de siempre: lee un dígito a la vez.</li>
          </ol>
        </div>
        <div className="flex flex-col gap-3">
          <h2 className="text-xl font-semibold">Cómo se entrenó</h2>
          <ul className="flex list-disc flex-col gap-2 pl-5 text-tinta-2">
            <li>Con MNIST (dígitos escritos a mano), con el optimizador Adam, también escrito a mano.</li>
            <li>Los gradientes se comprobaron contra diferencias finitas y el resultado se contrastó con la misma red en PyTorch.</li>
            <li>Con corrimientos y trazos más gruesos al entrenar, para tolerar dibujos hechos con el dedo.</li>
          </ul>
        </div>
      </section>

      <section className="rounded-2xl border border-linea p-4 text-tinta-2" aria-labelledby="limites">
        <h2 id="limites" className="mb-2 text-lg font-semibold text-tinta">
          Límites
        </h2>
        <p>
          La red solo conoce los dígitos del 0 al 9. Si dibujas una letra o un garabato, igual te responderá con un número. Se entrenó con un
          dígito por imagen, así que los dígitos de un número deben ir <strong>separados</strong>: si se tocan, el programa los ve como un solo
          trazo y avisa cuando lo nota (dos «1» pegados pueden pasar por uno solo). Funciona mejor con trazos grandes y claros. Es un proyecto
          de aprendizaje, no un producto.
        </p>
      </section>

      <footer className="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-linea pt-4 text-sm text-tinta-2">
        <a className="underline underline-offset-4 hover:text-tinta" href={REPOSITORIO} rel="noopener noreferrer" target="_blank">
          Código en GitHub
        </a>
        <span>Hecho por Johan Moreno · Ingeniería de Sistemas</span>
      </footer>
    </main>
  );
}
