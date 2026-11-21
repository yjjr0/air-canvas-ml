import { describe, expect, it } from 'vitest';
import { buildPrototype, cosine, l2normalize, matchCustom, type CustomGesture } from '../fewShot';

function rng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296 - 0.5;
  };
}


function samples(center: Float32Array, n: number, noise: number, seed: number): Float32Array[] {
  const r = rng(seed);
  return Array.from({ length: n }, () => l2normalize(center.map((v) => v + noise * r())));
}

const DIM = 32;
const axis = (i: number) => l2normalize(Float32Array.from({ length: DIM }, (_, j) => (j === i ? 1 : 0.05)));

const gesture = (proto: Float32Array, threshold: number): CustomGesture => ({
  id: 'g1',
  name: 'Thumbs up',
  action: 'none',
  prototype: Array.from(proto),
  threshold,
  samples: 5,
  createdAt: 0,
});

describe('few-shot prototypes', () => {
  it('averages samples into a unit prototype with a sensible threshold', () => {
    const center = axis(3);
    const { prototype, threshold } = buildPrototype(samples(center, 5, 0.2, 1));
    expect(Math.hypot(...prototype)).toBeCloseTo(1, 5);
    expect(cosine(prototype, center)).toBeGreaterThan(0.98);
    expect(threshold).toBeGreaterThanOrEqual(0.72);
    expect(threshold).toBeLessThanOrEqual(0.93);
  });

  it('loosens the threshold when the samples disagree', () => {
    const tight = buildPrototype(samples(axis(1), 5, 0.05, 2)).threshold;
    const loose = buildPrototype(samples(axis(1), 5, 0.6, 2)).threshold;
    expect(loose).toBeLessThan(tight);
  });

  it('matches the taught gesture and ignores unrelated poses', () => {
    const { prototype, threshold } = buildPrototype(samples(axis(5), 5, 0.15, 3));
    const g = gesture(prototype, threshold);
    const idleProto = axis(20);
    expect(matchCustom(samples(axis(5), 1, 0.15, 9)[0], [g], idleProto)?.gesture.id).toBe('g1');
    expect(matchCustom(axis(12), [g], idleProto)).toBeNull();
  });

  it('does not steal a trained class the embedding is closer to', () => {
    const pinch = axis(7);
    const taught = l2normalize(pinch.map((v, i) => v + (i === 8 ? 0.35 : 0)));
    const g = gesture(taught, 0.8);
    expect(cosine(pinch, taught)).toBeGreaterThan(0.8);
    expect(matchCustom(pinch, [g], pinch)).toBeNull();
  });
});
