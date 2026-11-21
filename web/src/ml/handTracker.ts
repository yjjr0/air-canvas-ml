import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { asset } from '../config';


const REMOTE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

export interface Tracker {
  landmarker: HandLandmarker;
  delegate: 'GPU' | 'CPU';
  modelSource: 'bundled' | 'remote';
}


export async function createTracker(): Promise<Tracker> {
  const [fileset, model] = await Promise.all([FilesetResolver.forVisionTasks(asset('mediapipe/wasm')), loadModel()]);
  const modelSource = model.source;
  let lastError: unknown;
  for (const delegate of ['GPU', 'CPU'] as const) {
    try {
      const landmarker = await HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer: model.bytes, delegate },
        runningMode: 'VIDEO',
        numHands: 1,
        minHandDetectionConfidence: 0.6,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      return { landmarker, delegate, modelSource };
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError instanceof Error ? lastError : new Error('Hand tracking could not start.');
}


async function loadModel(): Promise<{ bytes: Uint8Array; source: 'bundled' | 'remote' }> {
  const candidates = [
    [asset('models/hand_landmarker.task'), 'bundled'],
    [REMOTE_MODEL, 'remote'],
  ] as const;
  for (const [url, source] of candidates) {
    try {
      const res = await fetch(url);
      if (!res.ok || (res.headers.get('content-type') ?? '').includes('text/html')) continue;
      const bytes = new Uint8Array(await res.arrayBuffer());
      if (bytes.byteLength > 1_000_000) return { bytes, source };
    } catch {
    }
  }
  throw new Error('the hand landmark model could not be downloaded. Check your connection and try again.');
}
