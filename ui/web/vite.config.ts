import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

const apiPort = process.env.RENGU_FLOW_UI_PORT || "8765";
const devPort = Number(process.env.RENGU_FLOW_UI_DEV_PORT || "5173");

export default defineConfig({
  plugins: [vue()],
  base: "/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    host: "127.0.0.1",
    port: devPort,
    strictPort: true,
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${apiPort}`,
        changeOrigin: true,
        // Forward WebSocket upgrades (live progress + log streaming) to the API.
        ws: true,
      },
    },
  },
});
