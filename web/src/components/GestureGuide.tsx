import { GESTURES, READOUT_ORDER } from '../config';

export function GestureGuide() {
  return (
    <div className="guide">
      <dl className="gesture-list">
        {READOUT_ORDER.filter((l) => l !== 'idle').map((label) => (
          <div key={label}>
            <dt>
              {GESTURES[label].name}
              <span>{GESTURES[label].does}</span>
            </dt>
            <dd>{GESTURES[label].how}</dd>
          </div>
        ))}
      </dl>
      <p className="panel-foot">
        Keyboard: <kbd>Z</kbd> undo, <kbd>Shift</kbd>+<kbd>Z</kbd> redo, <kbd>H</kbd> show or hide the hand.
        Tips: draw in good lighting and using only one hand.
      </p>
    </div>
  );
}
