import { describe, expect, it } from 'vitest';
import { OneEuroFilter } from '../oneEuro';

describe('OneEuroFilter', () => {
  it('smooths jitter when the hand is still', () => {
    const f = new OneEuroFilter();
    let maxDev = 0;
    for (let i = 0; i < 120; i++) {
      const noisy = 0.5 + (i % 2 ? 0.01 : -0.01);
      const out = f.filter(noisy, i * 33);
      if (i > 30) maxDev = Math.max(maxDev, Math.abs(out - 0.5));
    }
    expect(maxDev).toBeLessThan(0.006);
  });

  it('keeps up with fast motion (low lag)', () => {
    const f = new OneEuroFilter();
    let out = 0;
    for (let i = 0; i < 30; i++) out = f.filter(i * 0.03, i * 33);
    expect(29 * 0.03 - out).toBeLessThan(0.05);
  });
});
