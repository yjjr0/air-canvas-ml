import { describe, expect, it } from 'vitest';
import fixtures from '../__fixtures__/parity.json';
import { FEATURES, WINDOW, median, windowFeatures } from '../normalize';

interface Case {
  label: string;
  aspect: number;
  frames: number[][];
  features: number[];
}

describe('windowFeatures matches the Python pipeline', () => {
  const cases = (fixtures as { cases: Case[]; window: number }).cases;

  it('has fixtures for every gesture', () => {
    expect((fixtures as { window: number }).window).toBe(WINDOW);
    expect(cases.length).toBeGreaterThanOrEqual(7);
  });

  for (const c of cases) {
    it(`${c.label}: features within 1e-5`, () => {
      const out = windowFeatures(c.frames, c.aspect);
      expect(out.length).toBe(FEATURES * WINDOW);
      let maxDiff = 0;
      for (let i = 0; i < out.length; i++) maxDiff = Math.max(maxDiff, Math.abs(out[i] - c.features[i]));
      expect(maxDiff).toBeLessThan(1e-5);
    });
  }
});

describe('feature invariances', () => {
  const base = (fixtures as { cases: Case[] }).cases[0];

  it('ignores where the hand is in the frame', () => {
    const shifted = base.frames.map((f) => f.map((v, i) => (i % 3 === 0 ? v + 0.1 : i % 3 === 1 ? v - 0.05 : v)));
    const a = windowFeatures(base.frames, base.aspect);
    const b = windowFeatures(shifted, base.aspect);
    for (let i = 0; i < a.length; i++) expect(b[i]).toBeCloseTo(a[i], 4);
  });

  it('ignores how far the hand is from the camera', () => {
    const cx = base.frames[0][0];
    const cy = base.frames[0][1];
    const scaled = base.frames.map((f) => f.map((v, i) => (i % 3 === 0 ? cx + (v - cx) * 1.6 : i % 3 === 1 ? cy + (v - cy) * 1.6 : v * 1.6)));
    const a = windowFeatures(base.frames, base.aspect);
    const b = windowFeatures(scaled, base.aspect);
    for (let i = 0; i < a.length; i++) expect(b[i]).toBeCloseTo(a[i], 4);
  });

  it('starts velocity at zero', () => {
    const out = windowFeatures(base.frames, base.aspect);
    for (let c = 63; c < FEATURES; c++) expect(out[c * WINDOW]).toBe(0);
  });
});

describe('median', () => {
  it('matches numpy for odd and even lengths', () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([])).toBe(0);
  });
});
