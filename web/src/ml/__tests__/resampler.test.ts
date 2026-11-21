import { describe, expect, it } from 'vitest';
import { Resampler } from '../resampler';

const frame = (v: number) => Float32Array.from({ length: 63 }, () => v);

describe('Resampler', () => {
  it('turns an irregular stream into exact 30 Hz frames by interpolation', () => {
    const r = new Resampler(30);
    expect(r.push(0, frame(0))).toBe(1);
    expect(r.push(100, frame(3))).toBe(3);
    const values = r.recent(4).map((f) => f[0]);
    expect(values[0]).toBe(0);
    expect(values[1]).toBeCloseTo(1, 5);
    expect(values[2]).toBeCloseTo(2, 5);
    expect(values[3]).toBeCloseTo(3, 5);
  });

  it('emits nothing between ticks and handles a fast tracker', () => {
    const r = new Resampler(30);
    r.push(0, frame(0));
    expect(r.push(10, frame(1))).toBe(0);
    expect(r.push(20, frame(2))).toBe(0);
    expect(r.push(40, frame(4))).toBe(1);
    expect(r.recent(1)[0][0]).toBeCloseTo(10 / 3, 4);
  });

  it('restarts instead of interpolating across a long gap', () => {
    const r = new Resampler(30, 64, 250);
    r.push(0, frame(0));
    r.push(100, frame(1));
    expect(r.length).toBe(4);
    expect(r.push(1000, frame(9))).toBe(1);
    expect(r.length).toBe(1);
    expect(r.recent(1)[0][0]).toBe(9);
  });

  it('pads the window by repeating the oldest frame while filling', () => {
    const r = new Resampler(30);
    r.push(0, frame(5));
    r.push(34, frame(6));
    const w = r.window(5);
    expect(w).toHaveLength(5);
    expect(w.map((f) => Math.round(f[0]))).toEqual([5, 5, 5, 5, 6]);
  });

  it('keeps at most `capacity` frames', () => {
    const r = new Resampler(30, 10);
    for (let i = 0; i <= 40; i++) r.push(i * (1000 / 30), frame(i));
    expect(r.length).toBe(10);
    expect(r.total).toBe(41);
  });
});
