import { useCallback, useEffect, useRef, useState } from 'react';
import type { AirCanvasEngine } from '../engine/engine';

export type RecorderPhase = 'idle' | 'countdown' | 'recording';

export interface RecorderState {
  phase: RecorderPhase;
  countdown: number;
  progress: number;
}

export type RecordResult = { ok: true; frames: Float32Array[] } | { ok: false; reason: string };

const NO_HAND_TIMEOUT_MS = 4000;

export function useRecorder(engine: AirCanvasEngine | null) {
  const [state, setState] = useState<RecorderState>({ phase: 'idle', countdown: 0, progress: 0 });
  const cancelRef = useRef<(() => void) | null>(null);

  useEffect(() => () => cancelRef.current?.(), []);

  const record = useCallback(
    (frames: number, countdownSec: number): Promise<RecordResult> =>
      new Promise((resolve) => {
        if (!engine) {
          resolve({ ok: false, reason: 'The model is still loading.' });
          return;
        }
        let cancelled = false;
        let timer: ReturnType<typeof setTimeout> | undefined;
        let unsubscribe: (() => void) | undefined;
        const finish = (result: RecordResult) => {
          cancelled = true;
          clearTimeout(timer);
          unsubscribe?.();
          cancelRef.current = null;
          setState({ phase: 'idle', countdown: 0, progress: 0 });
          resolve(result);
        };
        cancelRef.current = () => finish({ ok: false, reason: 'Recording cancelled.' });

        const startCapture = () => {
          if (cancelled) return;
          const captured: Float32Array[] = [];
          setState({ phase: 'recording', countdown: 0, progress: 0 });
          timer = setTimeout(() => finish({ ok: false, reason: 'No hand in view. Hold your hand up to the camera and try again.' }), NO_HAND_TIMEOUT_MS);
          unsubscribe = engine.onFrames((fresh, present) => {
            if (!present) {
              if (captured.length > 0) finish({ ok: false, reason: 'The hand left the frame. Try again.' });
              return;
            }
            captured.push(...fresh);
            setState({ phase: 'recording', countdown: 0, progress: Math.min(1, captured.length / frames) });
            if (captured.length >= frames) finish({ ok: true, frames: captured.slice(0, frames) });
          });
        };

        const tick = (left: number) => {
          if (cancelled) return;
          if (left <= 0) return startCapture();
          setState({ phase: 'countdown', countdown: left, progress: 0 });
          timer = setTimeout(() => tick(left - 1), 1000);
        };
        tick(countdownSec);
      }),
    [engine],
  );

  const cancel = useCallback(() => cancelRef.current?.(), []);

  return { state, record, cancel };
}
