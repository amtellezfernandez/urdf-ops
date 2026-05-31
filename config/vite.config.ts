import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { fileURLToPath } from "url";
import { runtimeUrls } from "./runtime.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");
const webRoot = path.resolve(rootDir, "web");
const iLoveUrdfSourceRoot = path.resolve(rootDir, "..", "i-love-urdf", "src");

const resolveClientApiBaseUrl = (mode: string): string =>
  process.env.URDF_OPS_API_BASE_URL ||
  process.env.VITE_API_BASE_URL ||
  (mode === "development" ? "/api" : runtimeUrls.apiBaseUrl);

export default defineConfig(({ mode }) => ({
  root: webRoot,
  cacheDir: path.resolve(rootDir, "node_modules", ".vite", "web"),
  server: {
    host: process.env.URDF_OPS_WEB_BIND_HOST || "127.0.0.1",
    port: Number(process.env.URDF_OPS_WEB_PORT || 5174),
    fs: {
      strict: true,
      allow: [
        webRoot,
        path.resolve(rootDir, "private_runtime", "src"),
        path.resolve(rootDir, "node_modules"),
        iLoveUrdfSourceRoot,
      ],
    },
    proxy: {
      "/api": {
        target: process.env.URDF_OPS_BACKEND_URL || "http://127.0.0.1:8001",
        changeOrigin: true,
        rewrite: (requestPath) => requestPath.replace(/^\/api/, ""),
      },
    },
    ...(mode === "test" || process.env.VITEST ? { hmr: false, ws: false } : {}),
  },
  plugins: [react()],
  define: {
    __URDF_CONFIG__: JSON.stringify({
      apiBaseUrl: resolveClientApiBaseUrl(mode),
      ikdBaseUrl: "",
      ikdWsUrl: "",
      teleopHttpBaseUrl: "",
      ikd: { enabled: false, useForDrag: false, controlHz: 0, telemetryHz: 0 },
      ik: {},
    }),
    "import.meta.env.VITE_BUILD_SHA": JSON.stringify(process.env.URDF_OPS_BUILD_SHA || "dev"),
  },
  css: { postcss: path.resolve(__dirname, "postcss.config.js") },
  resolve: {
    alias: {
      "@": path.resolve(webRoot, "src"),
      "@runtime-private": path.resolve(rootDir, "private_runtime", "src"),
      "i-love-urdf/browser": path.resolve(iLoveUrdfSourceRoot, "browser.ts"),
      "i-love-urdf": path.resolve(iLoveUrdfSourceRoot, "index.ts"),
    },
  },
  build: { chunkSizeWarningLimit: 900 },
}));
