import { defineConfig } from "@farm.js/core";

const nodeEnv =
  (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env ?? {};

function resolveAllowedApiOrigins(): string[] {
  const origins = new Set<string>(["http://127.0.0.1:8000", "http://localhost:8000"]);
  const rawApiBase = nodeEnv.VITE_API_BASE?.trim();
  if (rawApiBase) {
    try {
      const parsed = new URL(rawApiBase);
      if (parsed.protocol === "http:" || parsed.protocol === "https:") {
        origins.add(parsed.origin);
      }
    } catch {
      // Ignore invalid VITE_API_BASE values when deriving CSP origins
    }
  }
  return Array.from(origins);
}

const apiOrigins = resolveAllowedApiOrigins();
const isDev = nodeEnv.NODE_ENV !== "production";

export default defineConfig({
  devtools: false,
  theme: {
    default: "system",
  },
  images: {
    provider: "none",
  },
  docs: {
    enabled: false,
  },
  security: {
    csp: {
      nonce: true,
      directives: {
        "default-src": ["'self'"],
        "script-src": ["'self'"],
        "style-src": ["'self'", "'unsafe-inline'"],
        "img-src": ["'self'", "data:", "blob:", ...apiOrigins],
        "font-src": ["'self'", "data:"],
        "connect-src": [
          "'self'",
          ...apiOrigins,
          ...(isDev ? ["ws://localhost:*", "ws://127.0.0.1:*"] : []),
        ],
        "media-src": ["'self'", "blob:"],
        "object-src": ["'none'"],
        "base-uri": ["'self'"],
        "form-action": ["'self'"],
        "frame-ancestors": ["'none'"],
      },
    },
  },
  headers: () => [
    {
      source: "/*",
      headers: [
        { key: "X-Frame-Options", value: "DENY" },
        { key: "X-Content-Type-Options", value: "nosniff" },
        { key: "Referrer-Policy", value: "no-referrer" },
        {
          key: "Permissions-Policy",
          value: "camera=(), microphone=(), geolocation=()",
        },
      ],
    },
  ],
});
