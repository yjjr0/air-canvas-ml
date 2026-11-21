import type { InferenceSession, Tensor } from 'onnxruntime-web';
import { asset } from '../config';
import { FEATURES } from './normalize';

export type Backend = 'wasm' | 'webgpu';

export interface Prediction {
  probs: Float32Array;
  embedding: Float32Array;
  ms: number;
}

type OrtModule = typeof import('onnxruntime-web/wasm');

export class GestureClassifier {
  private constructor(
    private readonly ort: OrtModule,
    private readonly session: InferenceSession,
    readonly backend: Backend,
    readonly window: number,
  ) {}

  static async create(backend: Backend, window: number): Promise<GestureClassifier> {
    const ort: OrtModule = backend === 'webgpu' ? await import('onnxruntime-web/webgpu') : await import('onnxruntime-web/wasm');
    ort.env.wasm.wasmPaths = asset('ort/');
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = 'error';
    const session = await ort.InferenceSession.create(asset('models/gesture_cnn.onnx'), {
      executionProviders: [backend],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
    });
    const classifier = new GestureClassifier(ort, session, backend, window);
    await classifier.run(new Float32Array(FEATURES * window));
    return classifier;
  }

  static webgpuAvailable(): boolean {
    return typeof navigator !== 'undefined' && 'gpu' in navigator;
  }

  private queue: Promise<unknown> = Promise.resolve();

  run(features: Float32Array): Promise<Prediction> {
    const next = this.queue.then(() => this.runNow(features));
    this.queue = next.catch(() => undefined);
    return next;
  }

  private async runNow(features: Float32Array): Promise<Prediction> {
    const t0 = performance.now();
    const input: Tensor = new this.ort.Tensor('float32', features, [1, FEATURES, this.window]);
    const out = await this.session.run({ landmarks: input });
    const logits = (await out.logits.getData()) as Float32Array;
    const embedding = (await out.embedding.getData()) as Float32Array;
    const ms = performance.now() - t0;
    return { probs: softmax(logits), embedding: Float32Array.from(embedding), ms };
  }

  async dispose(): Promise<void> {
    await this.session.release();
  }
}

export function softmax(logits: ArrayLike<number>): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]);
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}
