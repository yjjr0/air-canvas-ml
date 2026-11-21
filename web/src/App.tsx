import { useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react';
import { AirCanvas } from './components/AirCanvas';
import { CollectorPanel } from './components/CollectorPanel';
import { GestureGuide } from './components/GestureGuide';
import { HowItWorks } from './components/HowItWorks';
import { ProbabilityStreaks } from './components/ProbabilityStreaks';
import { TeachPanel } from './components/TeachPanel';
import { TelemetryHUD } from './components/TelemetryHUD';
import { Toolbar } from './components/Toolbar';
import { REPO_URL } from './config';
import { AirCanvasEngine } from './engine/engine';
import { useEngineSnapshot } from './hooks/useEngine';

type Tab = 'gestures' | 'teach' | 'collect';
const TABS: { id: Tab; label: string }[] = [
  { id: 'gestures', label: 'Gestures' },
  { id: 'teach', label: 'Teach' },
  { id: 'collect', label: 'Collect data' },
];

const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const inkRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [engine, setEngine] = useState<AirCanvasEngine | null>(null);
  const snap = useEngineSnapshot(engine);
  const [reducedMotion] = useState(prefersReducedMotion);
  const [tab, setTab] = useState<Tab>(() => {
    const hash = window.location.hash.slice(1);
    return hash === 'teach' || hash === 'collect' ? hash : 'gestures';
  });

  useEffect(() => {
    const [video, ink, overlay] = [videoRef.current, inkRef.current, overlayRef.current];
    if (!video || !ink || !overlay) return;
    const instance = new AirCanvasEngine({ video, ink, overlay });
    setEngine(instance);
    let alive = true;
    void instance.init().then(() => {
      if (alive && !prefersReducedMotion()) void instance.startDemo();
    });
    return () => {
      alive = false;
      instance.dispose();
      setEngine(null);
    };
  }, []);

  useEffect(() => {
    if (!engine) return;
    const onKey = (ev: KeyboardEvent) => {
      if ((ev.target as HTMLElement).closest('input, select, textarea, [contenteditable="true"]')) return;
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
      const key = ev.key.toLowerCase();
      if (key === 'z') {
        if (ev.shiftKey) engine.redo();
        else engine.undo();
        ev.preventDefault();
      } else if (key === 'y') {
        engine.redo();
      } else if (key === 'h') {
        engine.toggleHand();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [engine]);

  const onTabKey = (ev: ReactKeyboardEvent<HTMLButtonElement>) => {
    const i = TABS.findIndex((t) => t.id === tab);
    const next = ev.key === 'ArrowRight' ? (i + 1) % TABS.length : ev.key === 'ArrowLeft' ? (i - 1 + TABS.length) % TABS.length : -1;
    if (next < 0) return;
    ev.preventDefault();
    setTab(TABS[next].id);
    document.getElementById(`tab-${TABS[next].id}`)?.focus();
  };

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <h1 className="wordmark">Air Canvas</h1>
          <p className="lede">
            Draw glowing strokes in mid-air using hand gestures!
            A custom gesture model trained in PyTorch and exported to ONNX running entirely in the browser.<br></br>
            Privacy Note: There is no server and the camera feed can be seen by you only.
          </p>
        </div>
        <a className="repo-link" href={REPO_URL}>
          Source on GitHub
        </a>
      </header>

      <main className="workspace">
        <section className="stage-column" aria-label="Canvas">
          <AirCanvas engine={engine} snap={snap} reducedMotion={reducedMotion} refs={{ video: videoRef, ink: inkRef, overlay: overlayRef }} />
          <Toolbar engine={engine} snap={snap} />
        </section>

        <aside className="side" aria-label="Model and tools">
          <ProbabilityStreaks snap={snap} />
          <TelemetryHUD engine={engine} snap={snap} />
          <section className="panel tabs-panel">
            <div className="tabs" role="tablist" aria-label="Tools">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  id={`tab-${t.id}`}
                  type="button"
                  role="tab"
                  aria-selected={tab === t.id}
                  aria-controls={`panel-${t.id}`}
                  tabIndex={tab === t.id ? 0 : -1}
                  className="tab"
                  onClick={() => setTab(t.id)}
                  onKeyDown={onTabKey}
                >
                  {t.label}
                </button>
              ))}
            </div>
            <div className="tabpanel" role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} tabIndex={0}>
              {tab === 'gestures' && <GestureGuide />}
              {tab === 'teach' && <TeachPanel engine={engine} snap={snap} />}
              {tab === 'collect' && <CollectorPanel engine={engine} snap={snap} />}
            </div>
          </section>
        </aside>
      </main>

      <HowItWorks meta={snap.meta} />

      <footer className="footer">
        <p>
          Air Canvas. No video, landmarks or drawing are uploaded anywhere.
        </p>
      </footer>
    </div>
  );
}
