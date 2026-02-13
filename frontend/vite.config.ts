import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

const devProxyTarget = process.env.VITE_DEV_PROXY_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  resolve: {
    dedupe: ["react", "react-dom"],
    alias: {
      react: path.resolve(__dirname, "node_modules/react"),
      "react-dom": path.resolve(__dirname, "node_modules/react-dom")
    }
  },
  server: {
    host: true,
    port: 3000,
    proxy: {
      "/api": {
        target: devProxyTarget,
        changeOrigin: true,
        ws: true
      }
    }
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    css: true
  }
});
