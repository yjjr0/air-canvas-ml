export const WINDOW = 20;
export const NUM_LANDMARKS = 21;
export const FEATURES = 126;
export const FRAME_SIZE = NUM_LANDMARKS * 3;

export type Frame = ArrayLike<number>;

export function windowFeatures(frames: readonly Frame[], aspect: number, out?: Float32Array): Float32Array {
  const T = frames.length;
  const result = out ?? new Float32Array(FEATURES * T);
  const p = new Float64Array(T * FRAME_SIZE);
  for (let t = 0; t < T; t++) {
    const f = frames[t];
    const o = t * FRAME_SIZE;
    for (let j = 0; j < NUM_LANDMARKS; j++) {
      p[o + j * 3] = f[j * 3] * aspect;
      p[o + j * 3 + 1] = f[j * 3 + 1];
      p[o + j * 3 + 2] = f[j * 3 + 2] * aspect;
    }
  }

  const palm = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    const o = t * FRAME_SIZE;
    const dx = p[o + 27] - p[o];
    const dy = p[o + 28] - p[o + 1];
    const dz = p[o + 29] - p[o + 2];
    palm[t] = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  const scale = Math.max(median(palm), 1e-6);

  for (let t = 0; t < T; t++) {
    const o = t * FRAME_SIZE;
    const prev = (t - 1) * FRAME_SIZE;
    for (let c = 0; c < FRAME_SIZE; c++) {
      const axis = c % 3;
      result[c * T + t] = (p[o + c] - p[o + axis]) / scale;
      result[(FRAME_SIZE + c) * T + t] = t > 0 ? (p[o + c] - p[prev + c]) / scale : 0;
    }
  }
  return result;
}

export function median(values: ArrayLike<number>): number {
  const sorted = Array.from(values).sort((a, b) => a - b);
  const n = sorted.length;
  if (n === 0) return 0;
  const mid = n >> 1;
  return n % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function toFrame(landmarks: ReadonlyArray<{ x: number; y: number; z: number }>, out = new Float32Array(FRAME_SIZE)): Float32Array {
  for (let j = 0; j < NUM_LANDMARKS; j++) {
    const l = landmarks[j];
    out[j * 3] = l.x;
    out[j * 3 + 1] = l.y;
    out[j * 3 + 2] = l.z;
  }
  return out;
}
