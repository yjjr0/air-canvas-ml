import { useSyncExternalStore } from 'react';
import type { AirCanvasEngine, EngineSnapshot } from '../engine/engine';

const noopSubscribe = () => () => undefined;
const EMPTY: EngineSnapshot = {
  mode: 'loading',
  status: 'Loading the gesture model…',
  error: null,
  meta: null,
  probs: null,
  customSims: {},
  customs: [],
  active: null,
  handPresent: false,
  trackerFps: 0,
  trackMs: 0,
  modelMs: 0,
  backend: 'wasm',
  backendBusy: false,
  delegate: null,
  inkIndex: 0,
  brush: 0.01,
  canUndo: false,
  canRedo: false,
  isEmpty: true,
  toast: null,
  showHand: true,
  aspect: 16 / 9,
};

export function useEngineSnapshot(engine: AirCanvasEngine | null): EngineSnapshot {
  return useSyncExternalStore(engine?.subscribe ?? noopSubscribe, engine?.getSnapshot ?? (() => EMPTY));
}
