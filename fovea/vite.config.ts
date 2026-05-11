import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Vite is the dev server + bundler; Tauri loads the built output (or
// devUrl during `tauri dev`).
//
// Notable: we deliberately fail fast on a stale port — Tauri picks the URL
// up from tauri.conf.json so changing it requires updating both files.
export default defineConfig(async () => ({
  plugins: [react()],

  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },

  // Tauri expects a fixed dev URL; if 1420 is taken we'd rather know.
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: {
      // Don't churn the dev server when the Rust side rebuilds.
      ignored: ["**/src-tauri/**"],
    },
  },
}));
