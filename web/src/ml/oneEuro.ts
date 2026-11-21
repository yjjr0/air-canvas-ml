export class OneEuroFilter {
  private x: number | null = null;
  private dx = 0;
  private t = 0;

  constructor(
    private readonly minCutoff = 1.4,
    private readonly beta = 8,
    private readonly dCutoff = 1.0,
  ) {}

  filter(value: number, tMs: number): number {
    if (this.x === null) {
      this.x = value;
      this.t = tMs;
      return value;
    }
    const dt = Math.max((tMs - this.t) / 1000, 1e-3);
    const rawDx = (value - this.x) / dt;
    const aD = smoothingAlpha(dt, this.dCutoff);
    this.dx = aD * rawDx + (1 - aD) * this.dx;
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    const a = smoothingAlpha(dt, cutoff);
    this.x = a * value + (1 - a) * this.x;
    this.t = tMs;
    return this.x;
  }

  reset(): void {
    this.x = null;
    this.dx = 0;
  }
}

function smoothingAlpha(dt: number, cutoff: number): number {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

export class OneEuro2D {
  private fx: OneEuroFilter;
  private fy: OneEuroFilter;

  constructor(minCutoff?: number, beta?: number) {
    this.fx = new OneEuroFilter(minCutoff, beta);
    this.fy = new OneEuroFilter(minCutoff, beta);
  }

  filter(x: number, y: number, tMs: number): { x: number; y: number } {
    return { x: this.fx.filter(x, tMs), y: this.fy.filter(y, tMs) };
  }

  reset(): void {
    this.fx.reset();
    this.fy.reset();
  }
}
