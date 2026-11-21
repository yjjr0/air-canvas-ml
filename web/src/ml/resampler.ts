export class Resampler {
  readonly dtMs: number;
  private frames: Float32Array[] = [];
  private last: { t: number; f: Float32Array } | null = null;
  private nextTick = 0;
  private emittedTotal = 0;

  constructor(
    readonly hz = 30,
    readonly capacity = 64,
    readonly maxGapMs = 250,
  ) {
    this.dtMs = 1000 / hz;
  }

  push(tMs: number, frame: ArrayLike<number>): number {
    if (!this.last || tMs - this.last.t > this.maxGapMs || tMs < this.last.t) {
      this.reset();
      const copy = Float32Array.from(frame);
      this.emit(copy);
      this.last = { t: tMs, f: copy };
      this.nextTick = tMs + this.dtMs;
      return 1;
    }
    if (tMs === this.last.t) return 0;
    const { t: t0, f: f0 } = this.last;
    let emitted = 0;
    while (this.nextTick <= tMs + 1e-6) {
      const a = (this.nextTick - t0) / (tMs - t0);
      const out = new Float32Array(f0.length);
      for (let i = 0; i < out.length; i++) out[i] = f0[i] + (frame[i] - f0[i]) * a;
      this.emit(out);
      this.nextTick += this.dtMs;
      emitted++;
    }
    this.last = { t: tMs, f: Float32Array.from(frame) };
    return emitted;
  }

  recent(n: number): Float32Array[] {
    return this.frames.slice(Math.max(0, this.frames.length - n));
  }

  window(size: number): Float32Array[] {
    const have = this.recent(size);
    if (have.length === 0) return [];
    const pad = size - have.length;
    return pad > 0 ? [...Array<Float32Array>(pad).fill(have[0]), ...have] : have;
  }

  get length(): number {
    return this.frames.length;
  }

  get total(): number {
    return this.emittedTotal;
  }

  reset(): void {
    this.frames = [];
    this.last = null;
  }

  private emit(frame: Float32Array): void {
    this.frames.push(frame);
    this.emittedTotal++;
    if (this.frames.length > this.capacity) this.frames.shift();
  }
}
