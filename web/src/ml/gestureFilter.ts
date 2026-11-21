import { FILTER } from '../config';

export interface FilterOptions {
  enterThreshold: number;
  exitThreshold: number;
  defaultEnterFrames: number;
  enterFrames: Record<string, number>;
  exitFrames: number;
}

export interface FilterUpdate {
  active: string | null;
  entered: string | null;
  exited: string | null;
}

export class GestureFilter {
  active: string | null = null;
  private candidate: string | null = null;
  private candCount = 0;
  private exitCount = 0;

  constructor(private readonly opts: FilterOptions = FILTER) {}

  update(scores: Readonly<Record<string, number>>): FilterUpdate {
    let entered: string | null = null;
    let exited: string | null = null;

    if (this.active !== null) {
      this.exitCount = (scores[this.active] ?? 0) < this.opts.exitThreshold ? this.exitCount + 1 : 0;
      if (this.exitCount >= this.opts.exitFrames) {
        exited = this.active;
        this.active = null;
        this.exitCount = 0;
      }
    }

    let top: string | null = null;
    let best = -Infinity;
    for (const [label, score] of Object.entries(scores)) {
      if (score > best) {
        best = score;
        top = label;
      }
    }

    if (top !== null && top !== this.active && best >= this.opts.enterThreshold) {
      if (top === this.candidate) this.candCount++;
      else {
        this.candidate = top;
        this.candCount = 1;
      }
      if (this.candCount >= (this.opts.enterFrames[top] ?? this.opts.defaultEnterFrames)) {
        if (this.active !== null) exited = this.active;
        this.active = top;
        entered = top;
        this.candidate = null;
        this.candCount = 0;
        this.exitCount = 0;
      }
    } else {
      this.candidate = null;
      this.candCount = 0;
    }
    return { active: this.active, entered, exited };
  }

  reset(): string | null {
    const was = this.active;
    this.active = null;
    this.candidate = null;
    this.candCount = 0;
    this.exitCount = 0;
    return was;
  }
}
