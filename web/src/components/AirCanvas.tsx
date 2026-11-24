import type { AirCanvasEngine, EngineSnapshot } from '../engine/engine';
import { GESTURES, type BaseLabel } from '../config';
import Maybe from '../maybe';
import type { RefObject } from 'react';
export interface StageRefs {
  video: RefObject<HTMLVideoElement | null>;
  ink: RefObject<HTMLCanvasElement | null>;
  overlay: RefObject<HTMLCanvasElement | null>;
}
interface Props {
  engine: AirCanvasEngine | null;
  snap: EngineSnapshot;
  reducedMotion: boolean;
  refs: StageRefs;
}

type StageCtaProps = {
  engine: AirCanvasEngine | null;
  snap: EngineSnapshot;
  live: boolean;
  busy: boolean;
  reducedMotion: boolean;
};

export function AirCanvas({ engine, snap, reducedMotion, refs }: Props) {
  const live = snap.mode === 'camera';
  const busy = snap.mode === 'loading' || snap.mode === 'camera-starting';

  const SOURCE_TEXT = new Map<EngineSnapshot['mode'], (snap: EngineSnapshot) => string>([
    ['demo', () => 'Live Demo'],
    ['camera', (snap) => (snap.handPresent ? 'Tracking your hand' : 'Hold one hand up to the camera')],
    ['camera-starting', () => 'Starting the camera'],
    ['loading', () => 'Loading'],
  ]);

  const sourceText = (snap: EngineSnapshot): string =>
    Maybe.of(SOURCE_TEXT.get(snap.mode))
      .map((format) => format(snap))
      .orElseGet('Paused');

  return (
    <div
      className="stage"
      style={{ aspectRatio: String(snap.aspect), maxWidth: `calc((100vh - 210px) * ${snap.aspect.toFixed(4)})` }}
      data-mode={snap.mode}
    >
      <video ref={refs.video} className="stage-video" playsInline muted aria-hidden="true" />
      <canvas ref={refs.ink} className="stage-layer" aria-label="Your drawing" role="img" />
      <canvas ref={refs.overlay} className="stage-layer" aria-hidden="true" />
      <span className="viewfinder" aria-hidden="true" />

      <div className="stage-status">
        <p className="source">{sourceText(snap)}</p>
        <ActiveGesture snap={snap} />
      </div>

      {snap.toast && (
        <p key={snap.toast.id} className="toast" role="status">
          {snap.toast.text}
        </p>
      )}

      <StageCta
        engine={engine}
        snap={snap}
        live={live}
        busy={busy}
        reducedMotion={reducedMotion}
      />

      {snap.error && (
        <p className="stage-error" role="alert">
          {snap.error}
        </p>
      )}
    </div>
  );
}

function ActiveGesture({ snap }: { snap: EngineSnapshot }) {
  const label = snap.active;

  if (!label || label === 'idle' || !snap.handPresent) return null;

  if (label.startsWith('custom:')) {
    const g = snap.customs.find((c) => `custom:${c.id}` === label);
    return (
      <p className="active-gesture">
        <span className="active-name">{g?.name ?? 'Taught gesture'}</span>
        <span className="active-does">taught gesture</span>
      </p>
    );
  }
  
  const info = GESTURES[label as BaseLabel];

  return (
    <p className="active-gesture">
      <span className="active-name">{info.name}</span>
      <span className="active-does">{info.does}</span>
    </p>
  );
}

function StageCta({ engine, snap, live, busy, reducedMotion }: StageCtaProps) {
  if (live) return null;

  if (busy) {
    return (
      <div className="stage-cta">
        <p className="stage-wait" role="status">
          {snap.status}
        </p>
      </div>
    );
  }

  return (
    <div className="stage-cta">
      <button
        type="button"
        className="button button-primary"
        onClick={() => void engine?.startCamera()}
        disabled={!engine || snap.mode === 'error'}
      >
        Use your camera
      </button>

      {snap.mode === 'ready' && (
        <button type="button" className="button" onClick={() => void engine?.startDemo()}>
          {reducedMotion ? 'Play the demo' : 'Replay the demo'}
        </button>
      )}

      <p className="stage-note">Video stays on this device. Nothing is uploaded.</p>
    </div>
  );
}