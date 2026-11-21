import { describe, expect, it } from 'vitest';
import stream from '../__fixtures__/stream.json';
import { GestureFilter } from '../gestureFilter';
import { FILTER } from '../../config';

const scores = (top: string, p: number, rest = 'idle') => ({ [top]: p, [rest]: 1 - p });

describe('GestureFilter', () => {
  it('needs consecutive confident frames before switching on', () => {
    const f = new GestureFilter();
    expect(f.update(scores('fist_erase', 0.9)).entered).toBeNull();
    expect(f.update(scores('fist_erase', 0.9)).entered).toBeNull();
    expect(f.update(scores('fist_erase', 0.9)).entered).toBe('fist_erase');
    expect(f.active).toBe('fist_erase');
  });

  it('lets pinch through faster: pen-down latency is what users feel', () => {
    const f = new GestureFilter();
    f.update(scores('pinch_draw', 0.8));
    expect(f.update(scores('pinch_draw', 0.8)).entered).toBe('pinch_draw');
  });

  it('asks for more evidence before the destructive clear', () => {
    const f = new GestureFilter();
    for (let i = 0; i < 3; i++) expect(f.update(scores('circle_clear', 0.95)).entered).toBeNull();
    expect(f.update(scores('circle_clear', 0.95)).entered).toBe('circle_clear');
  });

  it('holds through a brief confidence dip (hysteresis)', () => {
    const f = new GestureFilter();
    f.update(scores('pinch_draw', 0.9));
    f.update(scores('pinch_draw', 0.9));
    for (let i = 0; i < 5; i++) expect(f.update(scores('pinch_draw', 0.5, 'open_palm_hover')).active).toBe('pinch_draw');
    expect(f.update(scores('pinch_draw', 0.2, 'open_palm_hover')).exited).toBeNull();
    const out = f.update(scores('pinch_draw', 0.2, 'open_palm_hover'));
    expect(out.exited).toBe('pinch_draw');
    expect(out.active).toBeNull();
  });

  it('does not count interrupted streaks', () => {
    const f = new GestureFilter();
    f.update(scores('peace_color', 0.9));
    f.update(scores('peace_color', 0.9));
    f.update(scores('peace_color', 0.3, 'open_palm_hover'));
    expect(f.update(scores('peace_color', 0.9)).entered).toBeNull();
  });

  it('reproduces the events of ml/stream.py on the demo replay', () => {
    const { labels, probs, events } = stream as unknown as { labels: string[]; probs: number[][]; events: [number, string, string][] };
    expect(FILTER.exitFrames).toBe((stream as { exit_frames: number }).exit_frames);
    const f = new GestureFilter();
    const got: [number, string, string][] = [];
    probs.forEach((row, i) => {
      const s: Record<string, number> = {};
      labels.forEach((l, j) => (s[l] = row[j]));
      const { entered, exited } = f.update(s);
      if (exited) got.push([i, 'exit', exited]);
      if (entered) got.push([i, 'enter', entered]);
    });
    const expected = events.map(([t, kind, label]) => [Math.round(t * 30) - 7, kind, label]);
    expect(got).toEqual(expected);
    expect(got.filter(([, kind]) => kind === 'enter').length).toBeGreaterThan(10);
  });
});
