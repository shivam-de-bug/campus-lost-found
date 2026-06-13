import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:7860',
        changeOrigin: true,
      },
      '/report-found': {
        target: 'http://127.0.0.1:7860',
        changeOrigin: true,
      },
      '/search-lost': {
        target: 'http://127.0.0.1:7860',
        changeOrigin: true,
      },
      '/all-found': {
        target: 'http://127.0.0.1:7860',
        changeOrigin: true,
      },
      '/found_items': {
        target: 'http://127.0.0.1:7860',
        changeOrigin: true,
      }
    }
  }
})

