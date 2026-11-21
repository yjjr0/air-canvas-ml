import { BRUSH, INKS } from '../config';
import type { AirCanvasEngine, EngineSnapshot } from '../engine/engine';

export function Toolbar({ engine, snap }: { engine: AirCanvasEngine | null; snap: EngineSnapshot }) {
  const live = snap.mode === 'camera';
  return (
    <div className="toolbar" role="toolbar" aria-label="Drawing tools">
      <div className="inks" role="radiogroup" aria-label="Ink">
        {INKS.map((ink, i) => (
          <button
            key={ink.id}
            type="button"
            role="radio"
            aria-checked={snap.inkIndex === i}
            aria-label={ink.name}
            title={ink.name}
            className="ink"
            style={{ '--ink': ink.color } as React.CSSProperties}
            onClick={() => engine?.setInk(i)}
          />
        ))}
      </div>
      <label className="brush">
        <span>Brush</span>
        <input
          type="range"
          min={BRUSH.min}
          max={BRUSH.max}
          step={0.001}
          value={snap.brush}
          onChange={(e) => engine?.setBrush(Number(e.target.value))}
        />
      </label>
      <div className="tool-group">
        <button type="button" className="button" onClick={() => engine?.undo()} disabled={!snap.canUndo} aria-keyshortcuts="Z">
          Undo
        </button>
        <button type="button" className="button" onClick={() => engine?.redo()} disabled={!snap.canRedo} aria-keyshortcuts="Shift+Z">
          Redo
        </button>
        <button type="button" className="button" onClick={() => engine?.clear()} disabled={snap.isEmpty}>
          Clear
        </button>
        <button type="button" className="button" onClick={() => engine?.savePng()} disabled={snap.isEmpty}>
          Save PNG
        </button>
      </div>
      <div className="tool-group tool-end">
        <label className="check">
          <input type="checkbox" checked={snap.showHand} onChange={() => engine?.toggleHand()} />
          Show hand
        </label>
        {live && (
          <button type="button" className="button" onClick={() => engine?.stopCamera()}>
            Stop camera
          </button>
        )}
      </div>
    </div>
  );
}
