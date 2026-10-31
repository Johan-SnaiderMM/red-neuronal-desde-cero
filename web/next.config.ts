import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Todo el cálculo ocurre en el navegador: la página solo sirve archivos estáticos y el modelo (modelo.json).
  devIndicators: false,
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        ],
      },
      {
        // El modelo cambia de nombre solo si se reentrena: se puede cachear un buen rato, pero revalidando.
        source: "/modelo.json",
        headers: [{ key: "Cache-Control", value: "public, max-age=3600, stale-while-revalidate=86400" }],
      },
    ];
  },
};

export default nextConfig;
