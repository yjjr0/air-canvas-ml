import { useState } from 'react';
import type { AirCanvasEngine, EngineSnapshot } from '../engine/engine';
import { GestureClassifier, type Backend } from '../ml/onnxClassifier';

export function TelemetryHUD({ engine, snap }: { engine: AirCanvasEngine | null; snap: EngineSnapshot }) {
  const live = snap.mode === 'camera';
  const demo = snap.mode === 'demo';
  const webgpu = GestureClassifier.webgpuAvailable();
  const [pending, setPending] = useState<Backend | null>(null);
  const shown = pending ?? snap.backend;
  const choose = (backend: Backend) => {
    if (!engine || backend === snap.backend) return;
    setPending(backend);
    void engine.setBackend(backend).finally(() => setPending(null));
  };
  const fmt = (v: number, digits = 1) => (v > 0 ? v.toFixed(digits) : '–');

  return (
    <section className="panel telemetry" aria-labelledby="telemetry-title">
      <h2 id="telemetry-title">On this device</h2>
      <dl className="metrics">
        <div>
          <dt>Landmark frames</dt>
          <dd>{live || demo ? `${fmt(snap.trackerFps, 0)} per second` : '–'}</dd>
        </div>
        <div>
          <dt>Hand tracking</dt>
          <dd>{live ? `${fmt(snap.trackMs)} ms on ${snap.delegate === 'GPU' ? 'GPU' : 'CPU'}` : demo ? 'skipped in the replay' : '–'}</dd>
        </div>
        <div>
          <dt>Gesture model</dt>
          <dd>{snap.modelMs > 0 ? `${fmt(snap.modelMs, 2)} ms per window` : '–'}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd>{snap.meta ? `${(snap.meta.params / 1000).toFixed(1)}K parameters, ${Math.round(snap.meta.onnx_kb)} KB` : '–'}</dd>
        </div>
        <div>
          <dt>Input window</dt>
          <dd>{snap.meta ? `${snap.meta.window} frames at ${snap.meta.fps} Hz` : '–'}</dd>
        </div>
      </dl>
      <fieldset className="backend" disabled={!engine || snap.backendBusy} aria-busy={snap.backendBusy}>
        <legend>Run the model on{snap.backendBusy ? ' (switching…)' : ''}</legend>
        <label>
          <input type="radio" name="backend" checked={shown === 'wasm'} onChange={() => choose('wasm')} />
          CPU (WASM, 1 thread)
        </label>
        <label title={webgpu ? undefined : 'This browser does not expose WebGPU'}>
          <input type="radio" name="backend" checked={shown === 'webgpu'} disabled={!webgpu} onChange={() => choose('webgpu')} />
          GPU (WebGPU)
        </label>
      </fieldset>
    </section>
  );
}
