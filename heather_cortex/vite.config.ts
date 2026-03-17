import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/ui/',
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://localhost:6380',
      '/collections': 'http://localhost:6380',
    },
  },
})
