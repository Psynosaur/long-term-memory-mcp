import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// Backend API target — override with VITE_API_URL env var if the backend
// is running on a different host/port than the default 8666.
// e.g.: VITE_API_URL=http://192.168.1.5:8666 npm run dev
const API_TARGET = process.env.VITE_API_URL ?? 'http://127.0.0.1:8666'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      // Forward /api requests to the FastAPI backend during development.
      // In production the built SPA is served by FastAPI itself, so no proxy needed.
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
