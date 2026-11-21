import type { ActionId } from '../config';

export const SAMPLES_PER_GESTURE = 5;
export const MARGIN = 0.03;

export interface CustomGesture {
  id: string;
  name: string;
  action: ActionId;
  prototype: number[];
  threshold: number;
  samples: number;
  createdAt: number;
}

export interface CustomMatch {
  gesture: CustomGesture;
  similarity: number;
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function l2normalize(v: ArrayLike<number>): Float32Array {
  const out = Float32Array.from(v);
  const n = Math.sqrt(dot(out, out)) || 1;
  for (let i = 0; i < out.length; i++) out[i] /= n;
  return out;
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const na = Math.sqrt(dot(a, a)) || 1;
  const nb = Math.sqrt(dot(b, b)) || 1;
  return dot(a, b) / (na * nb);
}

export function buildPrototype(embeddings: ArrayLike<number>[]): { prototype: Float32Array; threshold: number; spread: number } {
  if (embeddings.length === 0) throw new Error('need at least one sample');
  const dim = embeddings[0].length;
  const mean = new Float32Array(dim);
  for (const e of embeddings) for (let i = 0; i < dim; i++) mean[i] += e[i] / embeddings.length;
  const prototype = l2normalize(mean);
  const sims = embeddings.map((e) => cosine(e, prototype));
  const avg = sims.reduce((a, b) => a + b, 0) / sims.length;
  const std = Math.sqrt(sims.reduce((a, b) => a + (b - avg) ** 2, 0) / sims.length);
  const threshold = clamp(Math.min(...sims) - 2 * std - 0.04, 0.72, 0.93);
  return { prototype, threshold, spread: 1 - avg };
}

export function similarities(embedding: ArrayLike<number>, gestures: readonly CustomGesture[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const g of gestures) out[g.id] = dot(embedding, g.prototype);
  return out;
}

export function matchCustom(
  embedding: ArrayLike<number>,
  gestures: readonly CustomGesture[],
  basePrototype: ArrayLike<number> | undefined,
): CustomMatch | null {
  let best: CustomMatch | null = null;
  for (const g of gestures) {
    const s = dot(embedding, g.prototype);
    if (s >= g.threshold && (!best || s > best.similarity)) best = { gesture: g, similarity: s };
  }
  if (!best) return null;
  const baseSim = basePrototype ? dot(embedding, basePrototype) : -1;
  return best.similarity >= baseSim + MARGIN ? best : null;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
