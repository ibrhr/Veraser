import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: ["cb55vllf-5173.thundercompute.net"],
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000"
    }
  }
});
