import type { Metadata, Viewport } from "next";
import "./globals.css";

const URL_SITIO = process.env.NEXT_PUBLIC_URL ?? "https://red-neuronal-desde-cero.vercel.app";
const TITULO = "Red neuronal desde cero · demo";
const DESCRIPCION =
  "Dibuja un dígito y una red neuronal hecha a mano con NumPy lo reconoce en tu navegador. Sin servidor: tu dibujo no sale de tu dispositivo.";

export const metadata: Metadata = {
  metadataBase: new URL(URL_SITIO),
  title: TITULO,
  description: DESCRIPCION,
  applicationName: "Red neuronal desde cero",
  // Vista previa al compartir el enlace (LinkedIn, WhatsApp): la imagen vive en `icons/`.
  openGraph: {
    type: "website",
    locale: "es_CO",
    siteName: "Red neuronal desde cero",
    title: TITULO,
    description: DESCRIPCION,
    images: [{ url: "/icons/og.png", width: 1200, height: 627, alt: "Red neuronal desde cero: reconoce dígitos dibujados, retropropagación escrita a mano" }],
  },
  twitter: { card: "summary_large_image", title: TITULO, description: DESCRIPCION, images: ["/icons/og.png"] },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f8fafc" },
    { media: "(prefers-color-scheme: dark)", color: "#0b1220" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
