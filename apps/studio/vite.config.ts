import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  assetsInclude: ["**/*.onnx"],
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  build: {
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/three") || id.includes("@react-three") || id.includes("postprocessing")) {
            return "three";
          }
          if (id.includes("node_modules/react") || id.includes("react-dom") || id.includes("zustand")) {
            return "react";
          }
          if (id.includes("jszip") || id.includes("zod") || id.includes("/packages/core/")) {
            return "core";
          }
          if (id.includes("d3-scale")) {
            return "d3";
          }
          if (id.includes("onnxruntime-web")) {
            return "onnx";
          }
          return undefined;
        },
      },
    },
  },
});
