import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// OPTIMUS_API points the dev proxy at another backend (default :8000)
const env = (globalThis as { process?: { env: Record<string, string | undefined> } }).process?.env ?? {}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: { '/api': env.OPTIMUS_API ?? 'http://127.0.0.1:8000' },
  },
})
