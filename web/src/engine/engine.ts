import { BRUSH, COOLDOWN_MS, FILTER, INKS, MIN_FRAMES, PEN_DOWN_BACKFILL_MS, PEN_UP_TRIM_MS, TRACK_LOST_MS, asset, type ActionId } from '../config';
import { InkDoc } from '../canvas/inkDoc';
import { InkRenderer, exportPng } from '../canvas/render';
import { matchCustom, similarities, type CustomGesture } from '../ml/fewShot';
import { GestureFilter } from '../ml/gestureFilter';
import type { Tracker } from '../ml/handTracker';
import { loadMeta, type GestureMeta } from '../ml/meta';
import { FEATURES, WINDOW, toFrame, windowFeatures } from '../ml/normalize';
import { GestureClassifier, type Backend, type Prediction } from '../ml/onnxClassifier';
import { OneEuro2D } from '../ml/oneEuro';
import { Resampler } from '../ml/resampler';
import { loadCustomGestures, saveCustomGestures } from '../storage/customGestures';

export type Mode = 'loading' | 'ready' | 'demo' | 'camera-starting' | 'camera' | 'error';

export interface EngineSnapshot {
  mode: Mode;
  status: string;
  error: string | null;
  meta: GestureMeta | null;
  probs: Float32Array | null;
  customSims: Record<string, number>;
  customs: CustomGesture[];
  active: string | null;
  handPresent: boolean;
  trackerFps: number;
  trackMs: number;
  modelMs: number;
  backend: Backend;
  backendBusy: boolean;
  delegate: string | null;
  inkIndex: number;
  brush: number;
  canUndo: boolean;
  canRedo: boolean;
  isEmpty: boolean;
  toast: { text: string; id: number } | null;
  showHand: boolean;
  aspect: number;
}

export interface StageElements {
  video: HTMLVideoElement;
  ink: HTMLCanvasElement;
  overlay: HTMLCanvasElement;
}

type FrameListener = (frames: Float32Array[], handPresent: boolean) => void;

const HAND_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];
const PALM_IDS = [0, 5, 9, 13, 17];
const DEMO_LOOP_PAUSE_MS = 2500;
const NIGHT = '#080a12';

const ema = (prev: number, next: number, a: number) => (prev === 0 ? next : prev + a * (next - prev));

export class AirCanvasEngine {
  readonly doc = new InkDoc();
  private listeners = new Set<() => void>();
  private frameListeners = new Set<FrameListener>();
  private snap: EngineSnapshot;
  private dirty = true;
  private lastPublish = 0;
  private toastSeq = 0;
  private toast: EngineSnapshot['toast'] = null;

  private meta: GestureMeta | null = null;
  private baseProtos: Record<string, number[]> = {};
  private classifier: GestureClassifier | null = null;
  private tracker: Tracker | null = null;
  private stream: MediaStream | null = null;
  private renderer: InkRenderer;

  private mode: Mode = 'loading';
  private status = 'Loading the gesture model…';
  private error: string | null = null;
  private aspect = 16 / 9;
  private resampler = new Resampler(30, 64, TRACK_LOST_MS);
  private filter = new GestureFilter(FILTER);
  private cursorFilter = new OneEuro2D(1.4, 8);
  private palmFilter = new OneEuro2D(1.0, 4);
  private latestFrame: Float32Array | null = null;
  private cursor: { x: number; y: number } | null = null;
  private cursorTrail: { x: number; y: number; t: number }[] = [];
  private palm: { x: number; y: number; r: number } | null = null;
  private lastSampleT = 0;
  private handPresent = false;
  private lastHandAt = 0;
  private active: string | null = null;
  private probs: Float32Array | null = null;
  private customSims: Record<string, number> = {};
  private customs: CustomGesture[] = loadCustomGestures();
  private inferring = false;
  private pendingInference = false;
  private cooldowns = new Map<string, number>();
  private featBuf = new Float32Array(FEATURES * WINDOW);
  private inkIndex = 0;
  private brush = BRUSH.initial;
  private showHand = true;
  private backend: Backend = 'wasm';
  private backendBusy = false;

  private trackMs = 0;
  private modelMs = 0;
  private trackerFps = 0;
  private lastTrackAt = 0;
  private lastVideoTime = -1;
  private detections = 0;
  private handednessLabel: string | null = null;

  private demo: { frames: Float32Array[]; fps: number; aspect: number } | null = null;
  private demoIdx = 0;
  private demoClock = 0;
  private demoEndedAt = -1;
  private demoEpoch = 0;
  private lastTick = 0;
  private raf = 0;
  private disposed = false;

  constructor(private readonly el: StageElements) {
    this.renderer = new InkRenderer(el.ink);
    this.snap = this.buildSnapshot();
    this.raf = requestAnimationFrame(this.tick);
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = (): EngineSnapshot => this.snap;

  onFrames(listener: FrameListener): () => void {
    this.frameListeners.add(listener);
    return () => this.frameListeners.delete(listener);
  }

  get currentAspect(): number {
    return this.aspect;
  }

  get handedness(): string | null {
    return this.mode === 'camera' ? this.handednessLabel : null;
  }

  async init(): Promise<void> {
    try {
      this.meta = await loadMeta();
      for (const [label, p] of Object.entries(this.meta.prototypes)) this.baseProtos[label] = p.vector;
      this.classifier = await GestureClassifier.create('wasm', WINDOW);
      this.setMode('ready', '');
    } catch (err) {
      this.fail(`The gesture model failed to load: ${messageOf(err)}`);
    }
  }

  async startDemo(): Promise<void> {
    if (!this.classifier) return;
    this.stopCamera();
    try {
      if (!this.demo) {
        const res = await fetch(asset('demo/demo_replay.json'));
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const doc = (await res.json()) as { frames: number[][]; fps: number; aspect: number };
        this.demo = { frames: doc.frames.map((f) => Float32Array.from(f)), fps: doc.fps, aspect: doc.aspect };
      }
    } catch (err) {
      this.fail(`The demo replay failed to load: ${messageOf(err)}`);
      return;
    }
    this.aspect = this.demo.aspect;
    this.restartDemoLoop();
    this.setMode('demo', '');
  }

  async startCamera(): Promise<void> {
    if (!this.classifier || this.mode === 'camera' || this.mode === 'camera-starting') return;
    this.stopDemo();
    this.error = null;
    this.setMode('camera-starting', 'Waiting for camera permission…');
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 960 }, height: { ideal: 540 }, frameRate: { ideal: 30, max: 60 } },
        audio: false,
      });
    } catch (err) {
      this.stream = null;
      this.error = cameraErrorText(err);
      this.setMode('ready', '');
      return;
    }
    const video = this.el.video;
    video.srcObject = this.stream;
    try {
      await video.play();
      this.aspect = video.videoWidth && video.videoHeight ? video.videoWidth / video.videoHeight : 4 / 3;
      if (!this.tracker) {
        this.setMode('camera-starting', 'Loading hand tracking (about 20 MB, cached after the first visit)…');
        const { createTracker } = await import('../ml/handTracker');
        this.tracker = await createTracker();
      }
    } catch (err) {
      this.stopCamera();
      this.fail(`Hand tracking could not start: ${messageOf(err)}`, 'ready');
      return;
    }
    this.resetSession();
    this.doc.reset();
    this.inkIndex = 0;
    this.setMode('camera', '');
  }

  stopCamera(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.el.video.srcObject = null;
    if (this.mode === 'camera' || this.mode === 'camera-starting') this.setMode('ready', '');
    this.resetSession();
  }

  async setBackend(backend: Backend): Promise<void> {
    if (backend === this.backend || this.backendBusy) return;
    this.backendBusy = true;
    this.markDirty();
    try {
      const next = await GestureClassifier.create(backend, WINDOW);
      const old = this.classifier;
      this.classifier = next;
      this.backend = backend;
      this.modelMs = 0;
      void old?.dispose();
      this.say(backend === 'webgpu' ? 'Gesture model now runs on WebGPU' : 'Gesture model now runs on WASM');
    } catch (err) {
      this.say(`WebGPU is not available here (${messageOf(err)}). Staying on WASM.`);
    } finally {
      this.backendBusy = false;
      this.markDirty();
    }
  }

  undo(): void {
    if (this.doc.undo()) this.say('Undid the last change');
    this.markDirty();
  }

  redo(): void {
    if (this.doc.redo()) this.say('Redid it');
    this.markDirty();
  }

  clear(): void {
    if (this.doc.clear()) this.say('Cleared. Swipe or press Z to undo.');
    this.markDirty();
  }

  nextInk(): void {
    this.setInk((this.inkIndex + 1) % INKS.length);
    this.say(`Ink: ${INKS[this.inkIndex].name}`);
  }

  setInk(index: number): void {
    this.inkIndex = index;
    this.markDirty();
  }

  setBrush(width: number): void {
    this.brush = Math.min(BRUSH.max, Math.max(BRUSH.min, width));
    this.markDirty();
  }

  toggleHand(): void {
    this.showHand = !this.showHand;
    this.markDirty();
  }

  savePng(): void {
    const h = 1080;
    const w = Math.round(h * this.aspect);
    const url = exportPng(this.doc, w, h, NIGHT);
    const a = document.createElement('a');
    a.href = url;
    a.download = `air-canvas-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.png`;
    a.click();
    this.say('Saved as PNG');
  }

  async embed(frames: Float32Array[]): Promise<Float32Array> {
    if (!this.classifier) throw new Error('model not loaded');
    const pred = await this.classifier.run(windowFeatures(frames, this.aspect));
    return pred.embedding;
  }

  addCustom(g: CustomGesture): void {
    this.customs = [...this.customs, g];
    saveCustomGestures(this.customs);
    this.markDirty();
  }

  removeCustom(id: string): void {
    this.customs = this.customs.filter((g) => g.id !== id);
    if (this.active === `custom:${id}`) this.filter.reset();
    saveCustomGestures(this.customs);
    this.markDirty();
  }

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.stopCamera();
    this.tracker?.landmarker.close();
    void this.classifier?.dispose();
    this.listeners.clear();
    this.frameListeners.clear();
  }

  private tick = (now: number): void => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.tick);
    const dt = this.lastTick ? Math.min(now - this.lastTick, 100) : 16;
    this.lastTick = now;
    if (this.mode === 'camera') this.pollCamera(now);
    else if (this.mode === 'demo') this.pollDemo(dt);
    if (this.handPresent && performance.now() - this.lastHandAt > TRACK_LOST_MS) this.loseHand();
    this.render();
    if (this.dirty && now - this.lastPublish > 66) this.publish(now);
  };

  private pollCamera(now: number): void {
    const video = this.el.video;
    if (!this.tracker || video.readyState < 2 || video.currentTime === this.lastVideoTime) return;
    this.lastVideoTime = video.currentTime;
    const t0 = performance.now();
    const result = this.tracker.landmarker.detectForVideo(video, now);
    if (++this.detections > 5) {
      this.trackMs = ema(this.trackMs, performance.now() - t0, 0.1);
      if (this.lastTrackAt) this.trackerFps = ema(this.trackerFps, 1000 / Math.max(1, now - this.lastTrackAt), 0.08);
    }
    this.lastTrackAt = now;
    const landmarks = result.landmarks[0];
    if (landmarks) {
      this.handednessLabel = result.handedness[0]?.[0]?.categoryName ?? null;
      this.ingest(toFrame(landmarks), now);
    }
    this.markDirty();
  }

  private pollDemo(dt: number): void {
    const demo = this.demo;
    if (!demo) return;
    this.demoClock += dt;
    const step = 1000 / demo.fps;
    const loopSpan = demo.frames.length * step + DEMO_LOOP_PAUSE_MS + 1000;
    while (this.demoIdx < demo.frames.length && this.demoIdx * step <= this.demoClock) {
      this.ingest(demo.frames[this.demoIdx], this.demoEpoch * loopSpan + this.demoIdx * step);
      this.demoIdx++;
    }
    this.trackerFps = demo.fps;
    this.trackMs = 0;
    if (this.demoIdx >= demo.frames.length) {
      if (this.demoEndedAt < 0) this.demoEndedAt = this.demoClock;
      else if (this.demoClock - this.demoEndedAt > DEMO_LOOP_PAUSE_MS) {
        this.demoEpoch++;
        this.restartDemoLoop();
      }
    }
  }

  private restartDemoLoop(): void {
    this.resetSession();
    this.doc.reset();
    this.inkIndex = 0;
    this.demoIdx = 0;
    this.demoClock = 0;
    this.demoEndedAt = -1;
  }

  private stopDemo(): void {
    if (this.mode === 'demo') this.setMode('ready', '');
    this.resetSession();
  }

  private ingest(frame: Float32Array, t: number): void {
    this.latestFrame = frame;
    this.lastSampleT = t;
    this.handPresent = true;
    this.lastHandAt = performance.now();

    this.cursor = this.cursorFilter.filter(1 - (frame[12] + frame[24]) / 2, (frame[13] + frame[25]) / 2, t);
    this.cursorTrail.push({ ...this.cursor, t });
    while (this.cursorTrail.length && this.cursorTrail[0].t < t - 400) this.cursorTrail.shift();
    let cx = 0;
    let cy = 0;
    for (const id of PALM_IDS) {
      cx += frame[id * 3] / PALM_IDS.length;
      cy += frame[id * 3 + 1] / PALM_IDS.length;
    }
    const palmLen = Math.hypot((frame[27] - frame[0]) * this.aspect, frame[28] - frame[1]);
    const smoothed = this.palmFilter.filter(1 - cx, cy, t);
    this.palm = { x: smoothed.x, y: smoothed.y, r: Math.max(0.02, 0.6 * palmLen) };

    const emitted = this.resampler.push(t, frame);
    if (emitted > 0) {
      const fresh = this.resampler.recent(emitted);
      this.frameListeners.forEach((l) => l(fresh, true));
      this.scheduleInference();
    }

    const op = this.doc.current;
    if (this.active === 'pinch_draw' && op?.kind === 'stroke') this.doc.addPoint({ ...this.cursor, t });
    else if (this.active === 'fist_erase' && op?.kind === 'erase') this.doc.addPoint({ x: this.palm.x, y: this.palm.y, t }, this.palm.r);
  }

  private loseHand(): void {
    this.handPresent = false;
    this.resampler.reset();
    const was = this.filter.reset();
    if (was) this.onExit(was);
    this.active = null;
    this.probs = null;
    this.cursorFilter.reset();
    this.palmFilter.reset();
    this.cursorTrail = [];
    this.frameListeners.forEach((l) => l([], false));
    this.markDirty();
  }

  private resetSession(): void {
    this.resampler.reset();
    const was = this.filter.reset();
    if (was) this.onExit(was);
    this.active = null;
    this.probs = null;
    this.handPresent = false;
    this.latestFrame = null;
    this.cursor = null;
    this.palm = null;
    this.cursorFilter.reset();
    this.palmFilter.reset();
    this.cursorTrail = [];
    this.lastVideoTime = -1;
    this.lastTrackAt = 0;
    this.detections = 0;
    this.trackMs = 0;
    this.trackerFps = 0;
    this.markDirty();
  }

  private scheduleInference(): void {
    if (this.inferring) {
      this.pendingInference = true;
      return;
    }
    const classifier = this.classifier;
    if (!classifier || this.resampler.length < MIN_FRAMES) return;
    this.inferring = true;
    windowFeatures(this.resampler.window(WINDOW), this.aspect, this.featBuf);
    classifier
      .run(this.featBuf)
      .then((pred) => {
        if (classifier === this.classifier && this.handPresent) this.onPrediction(pred);
      })
      .catch((err) => console.error('[air-canvas] inference failed', err))
      .finally(() => {
        this.inferring = false;
        if (this.pendingInference) {
          this.pendingInference = false;
          this.scheduleInference();
        }
      });
  }

  private onPrediction(pred: Prediction): void {
    const labels = this.meta?.labels ?? [];
    this.modelMs = ema(this.modelMs, pred.ms, 0.1);
    this.probs = pred.probs;
    const scores: Record<string, number> = {};
    let top = 0;
    labels.forEach((label, i) => {
      scores[label] = pred.probs[i];
      if (pred.probs[i] > pred.probs[top]) top = i;
    });
    if (this.customs.length) {
      this.customSims = similarities(pred.embedding, this.customs);
      const match = matchCustom(pred.embedding, this.customs, this.baseProtos[labels[top]]);
      if (match) {
        for (const label of labels) scores[label] = 0;
        scores[`custom:${match.gesture.id}`] = 1;
      }
    }
    const { entered, exited } = this.filter.update(scores);
    if (exited) this.onExit(exited);
    this.active = this.filter.active;
    if (entered) this.onEnter(entered);
    this.markDirty();
  }

  private onEnter(label: string): void {
    const t = this.lastSampleT;
    switch (label) {
      case 'pinch_draw': {
        const trail = this.cursorTrail.filter((p) => p.t >= t - PEN_DOWN_BACKFILL_MS);
        if (trail.length === 0 && this.cursor) trail.push({ ...this.cursor, t });
        if (trail.length === 0) break;
        this.doc.beginStroke(INKS[this.inkIndex].color, this.brush, trail[0]);
        for (const p of trail.slice(1)) this.doc.addPoint(p);
        break;
      }
      case 'fist_erase':
        if (this.palm) this.doc.beginErase(this.palm.r, { x: this.palm.x, y: this.palm.y, t });
        break;
      case 'peace_color':
        if (this.cooled(label)) this.nextInk();
        break;
      case 'swipe_undo':
        if (this.cooled(label)) this.undo();
        break;
      case 'circle_clear':
        if (this.cooled(label)) this.clear();
        break;
      default:
        if (label.startsWith('custom:')) {
          const gesture = this.customs.find((g) => `custom:${g.id}` === label);
          if (gesture && this.cooled(label, COOLDOWN_MS.custom)) this.runAction(gesture.action, gesture.name);
        }
    }
  }

  private onExit(label: string): void {
    const op = this.doc.current;
    if (label === 'pinch_draw' && op?.kind === 'stroke') this.doc.commit(this.lastSampleT - PEN_UP_TRIM_MS);
    if (label === 'fist_erase' && op?.kind === 'erase') this.doc.commit();
    this.markDirty();
  }

  private runAction(action: ActionId, name: string): void {
    switch (action) {
      case 'next_ink':
        return this.nextInk();
      case 'undo':
        return this.undo();
      case 'redo':
        return this.redo();
      case 'clear':
        return this.clear();
      case 'brush_up':
        this.setBrush(this.brush + BRUSH.step);
        return this.say(`${name}: thicker brush`);
      case 'brush_down':
        this.setBrush(this.brush - BRUSH.step);
        return this.say(`${name}: thinner brush`);
      case 'save_png':
        return this.savePng();
      default:
        return this.say(`Recognised “${name}”`);
    }
  }

  private cooled(key: string, ms = COOLDOWN_MS[key] ?? 800): boolean {
    const now = performance.now();
    if (now < (this.cooldowns.get(key) ?? 0)) return false;
    this.cooldowns.set(key, now + ms);
    return true;
  }

  private render(): void {
    const { ink, overlay } = this.el;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(ink.clientWidth * dpr);
    const h = Math.round(ink.clientHeight * dpr);
    if (w === 0 || h === 0) return;
    for (const c of [ink, overlay]) {
      if (c.width !== w || c.height !== h) {
        c.width = w;
        c.height = h;
      }
    }
    this.renderer.draw(this.doc);
    const ctx = overlay.getContext('2d')!;
    ctx.clearRect(0, 0, w, h);
    if (!this.handPresent || !this.latestFrame) return;
    const f = this.latestFrame;
    if (this.showHand) {
      ctx.lineWidth = Math.max(1, 1.25 * dpr);
      ctx.strokeStyle = 'rgba(239, 231, 218, 0.38)';
      ctx.beginPath();
      for (const [a, b] of HAND_EDGES) {
        ctx.moveTo((1 - f[a * 3]) * w, f[a * 3 + 1] * h);
        ctx.lineTo((1 - f[b * 3]) * w, f[b * 3 + 1] * h);
      }
      ctx.stroke();
      ctx.fillStyle = 'rgba(239, 231, 218, 0.7)';
      for (let j = 0; j < 21; j++) {
        ctx.beginPath();
        ctx.arc((1 - f[j * 3]) * w, f[j * 3 + 1] * h, 1.6 * dpr, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    const color = INKS[this.inkIndex].color;
    if (this.active === 'fist_erase' && this.palm) {
      ctx.setLineDash([4 * dpr, 4 * dpr]);
      ctx.strokeStyle = 'rgba(239, 231, 218, 0.85)';
      ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath();
      ctx.arc(this.palm.x * w, this.palm.y * h, this.palm.r * h, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    } else if (this.cursor) {
      const drawing = this.active === 'pinch_draw';
      const r = Math.max(4 * dpr, (this.brush * h) / 2 + 4 * dpr);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath();
      ctx.arc(this.cursor.x * w, this.cursor.y * h, r, 0, Math.PI * 2);
      ctx.stroke();
      if (drawing) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(this.cursor.x * w, this.cursor.y * h, Math.max(2, (this.brush * h) / 2), 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  private say(text: string): void {
    this.toast = { text, id: ++this.toastSeq };
    this.markDirty(true);
  }

  private fail(message: string, mode: Mode = 'error'): void {
    this.error = message;
    this.setMode(mode, '');
  }

  private setMode(mode: Mode, status: string): void {
    this.mode = mode;
    this.status = status;
    this.markDirty(true);
  }

  private markDirty(immediate = false): void {
    this.dirty = true;
    if (immediate) this.publish(performance.now());
  }

  private publish(now: number): void {
    this.snap = this.buildSnapshot();
    this.dirty = false;
    this.lastPublish = now;
    this.listeners.forEach((l) => l());
  }

  private buildSnapshot(): EngineSnapshot {
    return {
      mode: this.mode,
      status: this.status,
      error: this.error,
      meta: this.meta,
      probs: this.probs,
      customSims: this.customSims,
      customs: this.customs,
      active: this.active,
      handPresent: this.handPresent,
      trackerFps: this.trackerFps,
      trackMs: this.trackMs,
      modelMs: this.modelMs,
      backend: this.backend,
      backendBusy: this.backendBusy,
      delegate: this.tracker?.delegate ?? null,
      inkIndex: this.inkIndex,
      brush: this.brush,
      canUndo: this.doc.canUndo,
      canRedo: this.doc.canRedo,
      isEmpty: this.doc.isEmpty,
      toast: this.toast,
      showHand: this.showHand,
      aspect: this.aspect,
    };
  }
}

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function cameraErrorText(err: unknown): string {
  const name = err instanceof DOMException ? err.name : '';
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return 'Camera access is blocked. Allow it from the camera icon in the address bar, then try again.';
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'No camera was found on this device.';
  if (name === 'NotReadableError') return 'The camera is in use by another app. Close it and try again.';
  return `The camera could not start: ${messageOf(err)}`;
}
