import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('/node_modules/echarts/charts')) {
            return 'echarts-charts'
          }
          if (id.includes('/node_modules/echarts/components')) {
            return 'echarts-components'
          }
          if (id.includes('/node_modules/echarts/renderers')) {
            return 'echarts-renderers'
          }
          return undefined
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
