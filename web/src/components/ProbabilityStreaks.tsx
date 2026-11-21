import { GESTURES, READOUT_ORDER, type BaseLabel } from '../config';
import type { EngineSnapshot } from '../engine/engine';

export function ProbabilityStreaks({ snap }: { snap: EngineSnapshot }) {
  const labels = snap.meta?.labels ?? [];
  const value = (label: string): number => {
    const i = labels.indexOf(label);
    return snap.probs && i >= 0 ? snap.probs[i] : 0;
  };
  const waiting = !snap.probs;

  return (
    <section className="panel readout" aria-labelledby="readout-title">
      <h2 id="readout-title">What the model sees</h2>
      <p className="panel-sub">{waiting ? 'Probabilities appear once a hand is in view.' : 'Class probabilities, updated 30 times a second.'}</p>
      <ul className="streaks">
        {READOUT_ORDER.map((label) => {
          const p = value(label);
          const on = snap.active === label && snap.handPresent;
          return (
            <li key={label} className={on ? 'streak-row is-on' : 'streak-row'}>
              <span className="streak-name">{GESTURES[label as BaseLabel].name}</span>
              <span className="streak-does">{GESTURES[label as BaseLabel].does}</span>
              <span className="streak-track" aria-hidden="true">
                <span className="streak-fill" style={{ transform: `scaleX(${p.toFixed(3)})` }} />
              </span>
              <span className="streak-value">{waiting ? '–' : p.toFixed(2)}</span>
            </li>
          );
        })}
        {snap.customs.map((g) => {
          const s = snap.customSims[g.id] ?? 0;
          const on = snap.active === `custom:${g.id}` && snap.handPresent;
          const fill = Math.max(0, Math.min(1, (s - 0.5) / 0.5));
          return (
            <li key={g.id} className={on ? 'streak-row is-custom is-on' : 'streak-row is-custom'}>
              <span className="streak-name">{g.name}</span>
              <span className="streak-does">similarity, needs {g.threshold.toFixed(2)}</span>
              <span className="streak-track" aria-hidden="true">
                <span className="streak-fill" style={{ transform: `scaleX(${fill.toFixed(3)})` }} />
                <span className="streak-mark" style={{ left: `${((g.threshold - 0.5) / 0.5) * 100}%` }} />
              </span>
              <span className="streak-value">{waiting ? '–' : s.toFixed(2)}</span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
