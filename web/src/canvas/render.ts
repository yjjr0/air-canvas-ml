import type { InkDoc, InkOp, InkPoint } from './inkDoc';

export class InkRenderer {
  private layer: HTMLCanvasElement;
  private layerVersion = -1;
  private layerSize = '';

  constructor(private readonly target: HTMLCanvasElement) {
    this.layer = document.createElement('canvas');
  }

  draw(doc: InkDoc, glow = true): void {
    const { width, height } = this.target;
    const key = `${width}x${height}:${glow}`;
    if (doc.version !== this.layerVersion || key !== this.layerSize) {
      this.layer.width = width;
      this.layer.height = height;
      const lctx = this.layer.getContext('2d')!;
      lctx.clearRect(0, 0, width, height);
      for (const op of doc.ops) drawOp(lctx, op, width, height, glow);
      this.layerVersion = doc.version;
      this.layerSize = key;
    }
    const ctx = this.target.getContext('2d')!;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(this.layer, 0, 0);
    if (doc.current) drawOp(ctx, doc.current, width, height, glow);
  }
}

export function drawOp(ctx: CanvasRenderingContext2D, op: InkOp, w: number, h: number, glow: boolean): void {
  if (op.kind === 'clear') {
    ctx.clearRect(0, 0, w, h);
    return;
  }
  if (op.points.length === 0) return;
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  if (op.kind === 'erase') {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.strokeStyle = '#000';
    ctx.fillStyle = '#000';
    ctx.lineWidth = op.radius * 2 * h;
    tracePath(ctx, op.points, w, h);
    ctx.stroke();
    ctx.restore();
    return;
  }
  const lw = Math.max(1.5, op.width * h);
  if (glow) {
    ctx.globalCompositeOperation = 'lighter';
    ctx.shadowColor = op.color;
    ctx.shadowBlur = lw * 2.2;
    ctx.strokeStyle = op.color;
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = lw;
    tracePath(ctx, op.points, w, h);
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
    ctx.strokeStyle = mixWithWhite(op.color, 0.65);
    ctx.lineWidth = Math.max(1, lw * 0.36);
    tracePath(ctx, op.points, w, h);
    ctx.stroke();
  } else {
    ctx.strokeStyle = op.color;
    ctx.lineWidth = lw;
    tracePath(ctx, op.points, w, h);
    ctx.stroke();
  }
  ctx.restore();
}

function tracePath(ctx: CanvasRenderingContext2D, pts: InkPoint[], w: number, h: number): void {
  ctx.beginPath();
  const first = pts[0];
  ctx.moveTo(first.x * w, first.y * h);
  if (pts.length === 1) {
    ctx.lineTo(first.x * w + 0.01, first.y * h);
    return;
  }
  for (let i = 1; i < pts.length - 1; i++) {
    const p = pts[i];
    const n = pts[i + 1];
    ctx.quadraticCurveTo(p.x * w, p.y * h, ((p.x + n.x) / 2) * w, ((p.y + n.y) / 2) * h);
  }
  const last = pts[pts.length - 1];
  ctx.lineTo(last.x * w, last.y * h);
}

export function mixWithWhite(hex: string, amount: number): string {
  const v = parseInt(hex.slice(1), 16);
  const r = (v >> 16) & 255;
  const g = (v >> 8) & 255;
  const b = v & 255;
  const m = (c: number) => Math.round(c + (255 - c) * amount);
  return `rgb(${m(r)}, ${m(g)}, ${m(b)})`;
}

export function exportPng(doc: InkDoc, width: number, height: number, background: string): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ink = document.createElement('canvas');
  ink.width = width;
  ink.height = height;
  const ictx = ink.getContext('2d')!;
  for (const op of doc.ops) drawOp(ictx, op, width, height, true);
  if (doc.current) drawOp(ictx, doc.current, width, height, true);
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(ink, 0, 0);
  return canvas.toDataURL('image/png');
}
