import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

const jobs = [
  {
    from: 'node_modules/onnxruntime-web/dist',
    to: 'public/ort',
    files: [
      'ort-wasm-simd-threaded.mjs',
      'ort-wasm-simd-threaded.wasm',
      'ort-wasm-simd-threaded.asyncify.mjs',
      'ort-wasm-simd-threaded.asyncify.wasm',
    ],
  },
  {
    from: 'node_modules/@mediapipe/tasks-vision/wasm',
    to: 'public/mediapipe/wasm',
    files: ['vision_wasm_internal.js', 'vision_wasm_internal.wasm'],
  },
];

let copied = 0;
for (const job of jobs) {
  const dest = join(root, job.to);
  mkdirSync(dest, { recursive: true });
  for (const file of job.files) {
    const src = join(root, job.from, file);
    const out = join(dest, file);
    if (!existsSync(src)) {
      console.error(`[copy-runtime] missing ${job.from}/${file} - did npm install finish?`);
      process.exit(1);
    }
    if (existsSync(out) && statSync(out).size === statSync(src).size && statSync(out).mtimeMs >= statSync(src).mtimeMs) continue;
    copyFileSync(src, out);
    copied++;
  }
}
console.log(`[copy-runtime] ${copied ? `copied ${copied} file(s)` : 'up to date'}`);
