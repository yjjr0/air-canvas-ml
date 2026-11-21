import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { GESTURES, READOUT_ORDER, type BaseLabel } from '../config';
import type { AirCanvasEngine, EngineSnapshot } from '../engine/engine';
import { useRecorder } from '../hooks/useRecorder';
import { addClip, allClips, clearClips, deleteClip, exportDocument, type StoredClip } from '../storage/clipStore';

const CLIP_FRAMES = 30;
const SESSION = `s-${Date.now().toString(36)}`;
const round = (v: number) => Math.round(v * 1e5) / 1e5;

export function CollectorPanel({ engine, snap }: { engine: AirCanvasEngine | null; snap: EngineSnapshot }) {
  const [label, setLabel] = useState<BaseLabel>('pinch_draw');
  const [clips, setClips] = useState<StoredClip[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [batch, setBatch] = useState<{ done: number; total: number } | null>(null);
  const stopBatch = useRef(false);
  const { state, record, cancel } = useRecorder(engine);
  const live = snap.mode === 'camera';
  const busy = state.phase !== 'idle' || batch !== null;

  useEffect(() => {
    void allClips().then(setClips);
  }, []);

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const clip of clips) c[clip.label] = (c[clip.label] ?? 0) + 1;
    return c;
  }, [clips]);

  const recordOne = useCallback(
    async (countdown: number): Promise<boolean> => {
      if (!engine) return false;
      const result = await record(CLIP_FRAMES, countdown);
      if (!result.ok) {
        setMessage(result.reason);
        return false;
      }
      const clip: StoredClip = {
        label,
        frames: result.frames.map((f) => Array.from(f, round)),
        aspect: round(engine.currentAspect),
        handedness: engine.handedness,
        session: SESSION,
        t: Date.now(),
      };
      clip.id = await addClip(clip);
      setClips((prev) => [...prev, clip]);
      setMessage(`Saved a ${GESTURES[label].name.toLowerCase()} clip.`);
      return true;
    },
    [engine, label, record],
  );

  const recordBatch = async (n: number) => {
    stopBatch.current = false;
    setBatch({ done: 0, total: n });
    let done = 0;
    for (let i = 0; i < n && !stopBatch.current; i++) {
      if (await recordOne(i === 0 ? 3 : 1)) done++;
      setBatch({ done, total: n });
    }
    setBatch(null);
  };

  const exportJson = () => {
    const blob = new Blob([exportDocument(clips, SESSION)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `air-canvas-clips-${new Date().toISOString().slice(0, 10)}-${SESSION}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const removeLast = async () => {
    const last = clips[clips.length - 1];
    if (last?.id === undefined) return;
    await deleteClip(last.id);
    setClips((prev) => prev.slice(0, -1));
  };

  const removeAll = async () => {
    if (!window.confirm(`Delete all ${clips.length} recorded clips from this browser?`)) return;
    await clearClips();
    setClips([]);
  };

  return (
    <div className="collect">
      <p className="panel-sub">
        Record one-second clips to retrain the model using hand gestures. Clips are stored in this browser until you export them.
      </p>
      {!live && <p className="hint">Start your camera to record clips.</p>}
      <fieldset className="label-picker" disabled={busy}>
        <legend>Gesture to record</legend>
        {READOUT_ORDER.map((l) => (
          <label key={l} className={l === label ? 'is-picked' : undefined}>
            <input type="radio" name="clip-label" checked={l === label} onChange={() => setLabel(l)} />
            <span>{GESTURES[l].name}</span>
            <span className="count">{counts[l] ?? 0}</span>
          </label>
        ))}
      </fieldset>
      <p className="hint">{GESTURES[label].how} Vary distance, angle and lighting between clips.</p>
      {busy ? (
        <div className="recording" role="status">
          {batch && (
            <p className="recording-step">
              Clip {Math.min(batch.done + 1, batch.total)} of {batch.total}
            </p>
          )}
          <p className="recording-cue">{state.phase === 'countdown' ? `Get ready… ${state.countdown}` : state.phase === 'recording' ? 'Recording' : 'Next clip'}</p>
          <progress max={1} value={state.progress} />
          <button
            type="button"
            className="button"
            onClick={() => {
              stopBatch.current = true;
              cancel();
            }}
          >
            Stop
          </button>
        </div>
      ) : (
        <div className="tool-group">
          <button type="button" className="button button-primary" onClick={() => void recordOne(3)} disabled={!live}>
            Record a clip
          </button>
          <button type="button" className="button" onClick={() => void recordBatch(10)} disabled={!live}>
            Record 10 in a row
          </button>
        </div>
      )}
      {message && <p className="hint" role="status">{message}</p>}
      <div className="collect-foot">
        <p>
          {clips.length} {clips.length === 1 ? 'clip' : 'clips'} saved in this browser
        </p>
        <div className="tool-group">
          <button type="button" className="button" onClick={exportJson} disabled={clips.length === 0 || busy}>
            Export JSON
          </button>
          <button type="button" className="button button-quiet" onClick={() => void removeLast()} disabled={clips.length === 0 || busy}>
            Delete last
          </button>
          <button type="button" className="button button-quiet" onClick={() => void removeAll()} disabled={clips.length === 0 || busy}>
            Delete all
          </button>
        </div>
        <p className="panel-foot">
          Put the exported file in <code>ml/data/</code> and run <code>python ml/train_export.py</code>.
        </p>
      </div>
    </div>
  );
}
