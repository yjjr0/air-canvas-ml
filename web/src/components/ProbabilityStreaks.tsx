import type { EngineSnapshot } from '../engine/engine';
import { GESTURES, READOUT_ORDER, type BaseLabel } from '../config';
import Maybe from '../maybe';

type CustomGesture = EngineSnapshot['customs'][number];

interface StreakItem {
  key: string;
  name: string;
  does: string;
  fill: number;
  displayValue: string;
  active: boolean;
  custom?: boolean;
  markPct?: number;
}

const scaleSimilarity = (value: number): number =>
  Math.max(0, Math.min(1, (value - 0.5) / 0.5));

const probabilityFor = (snap: EngineSnapshot, label: string): number => {
    return Maybe.of(snap.meta?.labels)
    .map((labels) => labels.indexOf(label))
    .filter((i) => i >= 0)
    .map((i) => snap.probs?.[i])
    .orElseGet(0);
}

const toBaseStreak = (label: BaseLabel, snap: EngineSnapshot): StreakItem => {
  const p = probabilityFor(snap, label);
  const gesture = GESTURES[label];

  return {
    key: label,
    name: gesture.name,
    does: gesture.does,
    fill: p,
    displayValue: snap.probs ? p.toFixed(2) : '–',
    active: snap.active === label && snap.handPresent,
  };
}

const toCustomStreak = (gesture: CustomGesture, snap: EngineSnapshot): StreakItem  => {
  const s = snap.customSims[gesture.id] ?? 0;

  return {
    key: gesture.id,
    name: gesture.name,
    does: `similarity, needs ${gesture.threshold.toFixed(2)}`,
    fill: scaleSimilarity(s),
    displayValue: snap.probs ? s.toFixed(2) : '–',
    active: snap.active === `custom:${gesture.id}` && snap.handPresent,
    custom: true,
    markPct: ((gesture.threshold - 0.5) / 0.5) * 100,
  };
}

export function ProbabilityStreaks({ snap }: { snap: EngineSnapshot }) {
  const waiting = !snap.probs;
  const streaks: StreakItem[] = [
    ...READOUT_ORDER.map((label) => toBaseStreak(label, snap)),
    ...snap.customs.map((gesture) => toCustomStreak(gesture, snap)),
  ];

  return (
    <section className="panel readout" aria-labelledby="readout-title">
      <h2 id="readout-title">What the model sees</h2>
      <p className="panel-sub">
        {waiting
          ? 'Probabilities appear once a hand is in view.'
          : 'Class probabilities, updated 30 times a second.'}
      </p>
      <ul className="streaks">
        {streaks.map((item) => (
          <StreakRow key={item.key} item={item} />
        ))}
      </ul>
    </section>
  );
}

function StreakRow({ item }: { item: StreakItem }) {
  const className = [
    'streak-row',
    item.custom && 'is-custom',
    item.active && 'is-on',
  ].filter(Boolean)
  .join(' ');

  return (
    <li className={className}>
      <span className="streak-name">{item.name}</span>
      <span className="streak-does">{item.does}</span>
      <span className="streak-track" aria-hidden="true">
        <span className="streak-fill" style={{ transform: `scaleX(${item.fill.toFixed(3)})` }} />
        {item.markPct !== undefined && (
          <span className="streak-mark" style={{ left: `${item.markPct}%` }} />
        )}
      </span>
      <span className="streak-value">{item.displayValue}</span>
    </li>
  );
}