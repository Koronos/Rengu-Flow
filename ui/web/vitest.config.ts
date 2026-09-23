import { defineConfig } from "vitest/config";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  test: {
    environment: "happy-dom",
    include: ["src/**/*.test.ts"],
    // ponytail: drawer tests mount the full stage forms (~5 s on this box); 5 s default flaked.
    testTimeout: 20_000,
  },
});
