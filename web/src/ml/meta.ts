import { asset } from '../config';

export interface EvalMetrics {
  n: number;
  accuracy: number;
  macro_f1: number;
}

export interface GestureMeta {
  model: string;
  labels: string[];
  static_labels: string[];
  motion_labels: string[];
  input: { name: string; shape: number[] };
  fps: number;
  window: number;
  feature_version: number;
  embedding_dim: number;
  params: number;
  onnx_kb: number;
  prototypes: Record<string, { vector: number[]; sim_p05: number }>;
  trained_at: string;
  data: { synthetic_clips: number; real_clips: number; real_sessions: number };
  metrics: Record<string, EvalMetrics>;
}

export const SUPPORTED_FEATURE_VERSION = 1;

export async function loadMeta(): Promise<GestureMeta> {
  const res = await fetch(asset('models/gesture_meta.json'));
  if (!res.ok) throw new Error(`Could not load the model description (HTTP ${res.status}).`);
  const meta = (await res.json()) as GestureMeta;
  if (meta.feature_version !== SUPPORTED_FEATURE_VERSION) {
    throw new Error(`Model expects feature version ${meta.feature_version}; this build computes version ${SUPPORTED_FEATURE_VERSION}.`);
  }
  return meta;
}
