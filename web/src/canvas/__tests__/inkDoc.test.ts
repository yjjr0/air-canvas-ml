import { describe, expect, it } from 'vitest';
import { InkDoc } from '../inkDoc';

const pt = (x: number, t: number) => ({ x, y: 0.5, t });

function stroke(doc: InkDoc, from: number, n = 5) {
  doc.beginStroke('#fff', 0.01, pt(from, 0));
  for (let i = 1; i < n; i++) doc.addPoint(pt(from + i * 0.01, i * 33));
  doc.commit();
}

describe('InkDoc', () => {
  it('undoes and redoes strokes', () => {
    const doc = new InkDoc();
    stroke(doc, 0.1);
    stroke(doc, 0.5);
    expect(doc.ops).toHaveLength(2);
    expect(doc.undo()).toBe(true);
    expect(doc.ops).toHaveLength(1);
    expect(doc.redo()).toBe(true);
    expect(doc.ops).toHaveLength(2);
  });

  it('makes clear undoable', () => {
    const doc = new InkDoc();
    stroke(doc, 0.1);
    expect(doc.clear()).toBe(true);
    expect(doc.isEmpty).toBe(true);
    doc.undo();
    expect(doc.isEmpty).toBe(false);
  });

  it('does not record a clear of an empty canvas', () => {
    const doc = new InkDoc();
    expect(doc.clear()).toBe(false);
    expect(doc.ops).toHaveLength(0);
  });

  it('trims the pen-up tail of a stroke', () => {
    const doc = new InkDoc();
    doc.beginStroke('#fff', 0.01, pt(0, 0));
    for (let i = 1; i <= 10; i++) doc.addPoint(pt(i * 0.01, i * 33));
    doc.commit(33 * 7);
    const op = doc.ops[0];
    expect(op.kind === 'stroke' && op.points.length).toBe(8);
  });

  it('drops sub-pixel jitter points', () => {
    const doc = new InkDoc();
    doc.beginStroke('#fff', 0.01, pt(0.5, 0));
    doc.addPoint(pt(0.5001, 10));
    doc.addPoint(pt(0.52, 20));
    expect(doc.current?.points).toHaveLength(2);
  });

  it('a new stroke clears the redo stack', () => {
    const doc = new InkDoc();
    stroke(doc, 0.1);
    doc.undo();
    stroke(doc, 0.3);
    expect(doc.canRedo).toBe(false);
  });
});
