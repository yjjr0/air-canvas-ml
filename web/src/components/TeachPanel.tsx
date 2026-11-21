import { useState } from 'react';
import { ACTIONS, type ActionId } from '../config';
import type { AirCanvasEngine, EngineSnapshot } from '../engine/engine';
import { SAMPLES_PER_GESTURE, buildPrototype } from '../ml/fewShot';
import { WINDOW } from '../ml/normalize';
import { useRecorder } from '../hooks/useRecorder';

const SAMPLE_FRAMES = 24; 

export function TeachPanel({ engine, snap }: { engine: AirCanvasEngine | null; snap: EngineSnapshot }) {
  const [name, setName] = useState('');
  const [action, setAction] = useState<ActionId>('none');
  const [taken, setTaken] = useState(0);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const { state, record, cancel } = useRecorder(engine);
  const live = snap.mode === 'camera';

  const teach = async () => {
    if (!engine) return;
    const label = name.trim();
    if (!label) {
      setMessage('Give the gesture a name first.');
      return;
    }
    setBusy(true);
    setMessage(null);
    const embeddings: Float32Array[] = [];
    let failures = 0;
    try {
      while (embeddings.length < SAMPLES_PER_GESTURE) {
        setTaken(embeddings.length);
        const result = await record(SAMPLE_FRAMES, embeddings.length === 0 ? 3 : 2);
        if (!result.ok) {
          setMessage(result.reason);
          if (result.reason === 'Recording cancelled.' || ++failures > 3) return;
          continue;
        }
        embeddings.push(await engine.embed(result.frames.slice(-WINDOW)));
      }
      const { prototype, threshold } = buildPrototype(embeddings);
      engine.addCustom({
        id: crypto.randomUUID(),
        name: label,
        action,
        prototype: Array.from(prototype),
        threshold,
        samples: embeddings.length,
        createdAt: Date.now(),
      });
      setMessage(`Learned “${label}”. Try it: its row appears in the readout above.`);
      setName('');
    } finally {
      setBusy(false);
      setTaken(0);
    }
  };

  return (
    <div className="teach">
      <p className="panel-sub">
        Record a new hand gesture or motion five times. The model compares its 32-number embedding of what it sees with
        the average of your samples. It learns instantly without retraining.
      </p>
      {!live && <p className="hint">Start your camera to teach a gesture.</p>}
      <div className="form-row">
        <label>
          <span>Name</span>
          <input type="text" value={name} maxLength={24} placeholder="Thumbs up" onChange={(e) => setName(e.target.value)} disabled={busy} />
        </label>
        <label>
          <span>When it is recognised</span>
          <select value={action} onChange={(e) => setAction(e.target.value as ActionId)} disabled={busy}>
            {Object.entries(ACTIONS).map(([id, text]) => (
              <option key={id} value={id}>
                {text}
              </option>
            ))}
          </select>
        </label>
      </div>
      {busy ? (
        <div className="recording" role="status">
          <p className="recording-step">
            Sample {taken + 1} of {SAMPLES_PER_GESTURE}
          </p>
          <p className="recording-cue">
            {state.phase === 'countdown' ? `Get ready… ${state.countdown}` : state.phase === 'recording' ? 'Hold it' : 'Next sample'}
          </p>
          <progress max={1} value={state.progress} />
          <button type="button" className="button" onClick={cancel}>
            Cancel
          </button>
        </div>
      ) : (
        <button type="button" className="button button-primary" onClick={() => void teach()} disabled={!live}>
          Record {SAMPLES_PER_GESTURE} samples
        </button>
      )}
      {message && <p className="hint" role="status">{message}</p>}
      {snap.customs.length > 0 && (
        <ul className="taught">
          {snap.customs.map((g) => (
            <li key={g.id}>
              <span className="taught-name">{g.name}</span>
              <span className="taught-action">{ACTIONS[g.action]}</span>
              <button type="button" className="button button-quiet" onClick={() => engine?.removeCustom(g.id)} aria-label={`Forget ${g.name}`}>
                Forget
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
