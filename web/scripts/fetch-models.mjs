import { existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dest = join(root, 'public/models/hand_landmarker.task');

if (existsSync(dest) && statSync(dest).size > 1_000_000) {
  console.log('[fetch-models] hand_landmarker.task already present');
  process.exit(0);
}

try {
  const res = await fetch(URL);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const bytes = Buffer.from(await res.arrayBuffer());
  if (bytes.length < 1_000_000) throw new Error(`unexpectedly small file (${bytes.length} bytes)`);
  mkdirSync(dirname(dest), { recursive: true });
  writeFileSync(dest, bytes);
  console.log(`[fetch-models] saved hand_landmarker.task (${(bytes.length / 1e6).toFixed(1)} MB)`);
} catch (err) {
  console.warn(`[fetch-models] could not download the hand model (${err.message}); the app will load it from Google's CDN at runtime.`);
}
