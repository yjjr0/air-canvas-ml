/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { join, normalize } from 'node:path';
import { defaultClientConditions, defineConfig, type Plugin } from 'vite';

const base = process.env.VITE_BASE ?? '/air-canvas-ml/';

function serveRuntimeFiles(): Plugin {
  const dirs = ['ort/', 'mediapipe/wasm/'];
  return {
    name: 'serve-runtime-files',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = decodeURIComponent((req.url ?? '').split('?')[0]);
        for (const dir of dirs) {
          for (const prefix of [`${base}${dir}`, `/${dir}`]) {
            if (!path.startsWith(prefix)) continue;
            const root = join(server.config.publicDir, dir);
            const file = normalize(join(root, path.slice(prefix.length)));
            if (file.startsWith(root) && existsSync(file) && statSync(file).isFile()) {
              res.setHeader('Content-Type', file.endsWith('.wasm') ? 'application/wasm' : 'text/javascript');
              createReadStream(file).pipe(res);
              return;
            }
          }
        }
        next();
      });
    },
  };
}

export default defineConfig({
  base,
  plugins: [react(), serveRuntimeFiles()],
  resolve: {
    conditions: [...defaultClientConditions, 'onnxruntime-web-use-extern-wasm'],
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  build: {
    target: 'es2022',
    chunkSizeWarningLimit: 700,
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
