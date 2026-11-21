export interface InkPoint {
  x: number;
  y: number;
  t: number;
}

export type InkOp =
  | { kind: 'stroke'; color: string; width: number; points: InkPoint[] }
  | { kind: 'erase'; radius: number; points: InkPoint[] }
  | { kind: 'clear' };

const MIN_STEP = 0.0015;

export class InkDoc {
  ops: InkOp[] = [];
  current: Extract<InkOp, { points: InkPoint[] }> | null = null;
  private redoStack: InkOp[] = [];
  version = 0;

  beginStroke(color: string, width: number, at: InkPoint): void {
    this.commit();
    this.current = { kind: 'stroke', color, width, points: [at] };
  }

  beginErase(radius: number, at: InkPoint): void {
    this.commit();
    this.current = { kind: 'erase', radius, points: [at] };
  }

  addPoint(p: InkPoint, radius?: number): void {
    const op = this.current;
    if (!op) return;
    if (op.kind === 'erase' && radius !== undefined) op.radius = Math.max(op.radius, radius);
    const last = op.points[op.points.length - 1];
    if (Math.hypot(p.x - last.x, p.y - last.y) < MIN_STEP) return;
    op.points.push(p);
  }

  commit(trimAfter?: number): void {
    const op = this.current;
    this.current = null;
    if (!op) return;
    if (trimAfter !== undefined && op.points.length > 2) {
      const kept = op.points.filter((p) => p.t <= trimAfter);
      op.points = kept.length >= 2 ? kept : op.points.slice(0, 2);
    }
    this.ops.push(op);
    this.redoStack = [];
    this.version++;
  }

  undo(): boolean {
    this.commit();
    const op = this.ops.pop();
    if (!op) return false;
    this.redoStack.push(op);
    this.version++;
    return true;
  }

  redo(): boolean {
    this.commit();
    const op = this.redoStack.pop();
    if (!op) return false;
    this.ops.push(op);
    this.version++;
    return true;
  }

  clear(): boolean {
    this.commit();
    if (this.isEmpty) return false;
    this.ops.push({ kind: 'clear' });
    this.redoStack = [];
    this.version++;
    return true;
  }

  reset(): void {
    this.ops = [];
    this.redoStack = [];
    this.current = null;
    this.version++;
  }

  get canUndo(): boolean {
    return this.ops.length > 0 || this.current !== null;
  }

  get canRedo(): boolean {
    return this.redoStack.length > 0;
  }

  get isEmpty(): boolean {
    for (let i = this.ops.length - 1; i >= 0; i--) {
      const op = this.ops[i];
      if (op.kind === 'clear') return true;
      if (op.kind === 'stroke') return false;
    }
    return this.current?.kind !== 'stroke';
  }
}
