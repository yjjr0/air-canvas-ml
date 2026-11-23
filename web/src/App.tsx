import { AirCanvas } from './components/AirCanvas';
import { AirCanvasEngine } from './engine/engine';
import { CollectorPanel } from './components/CollectorPanel';
import { GestureGuide } from './components/GestureGuide';
import { HowItWorks } from './components/HowItWorks';
import Maybe from './maybe';
import { ProbabilityStreaks } from './components/ProbabilityStreaks';
import { REPO_URL } from './config';
import { TeachPanel } from './components/TeachPanel';
import { TelemetryHUD } from './components/TelemetryHUD';
import { Toolbar } from './components/Toolbar';
import { useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from 'react';
import { useEngineSnapshot } from './hooks/useEngine';

type Tab = 'gestures' | 'teach' | 'collect';
const TABS: { id: Tab; label: string }[] = [
  { id: 'gestures', label: 'Gestures' },
  { id: 'teach', label: 'Teach' },
  { id: 'collect', label: 'Collect data' },
];

const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const changeTab = () => {
    const hash = window.location.hash.slice(1);
    return hash === 'teach' || hash === 'collect' ? hash : 'gestures';
}

export function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const inkRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [engine, setEngine] = useState<AirCanvasEngine | null>(null);
  const [reducedMotion] = useState(prefersReducedMotion);
  const [tab, setTab] = useState<Tab>(changeTab);
  const snap = useEngineSnapshot(engine);

  const onTabKey = (ev: ReactKeyboardEvent<HTMLButtonElement>) => {
    const i = TABS.findIndex((t) => t.id === tab);
    const next = ev.key === 'ArrowRight' ? (i + 1) % TABS.length : ev.key === 'ArrowLeft' ? (i - 1 + TABS.length) % TABS.length : -1;
    if (next < 0) return;
    ev.preventDefault();
    setTab(TABS[next].id);
    document.getElementById(`tab-${TABS[next].id}`)?.focus();
  };
  
  const engineLifecycle = () =>
    Maybe.of({ video: videoRef.current, ink: inkRef.current, overlay: overlayRef.current })
      .map(({ video, ink, overlay }) => video && ink && overlay ? { video, ink, overlay } : null)
      .map((elements) => {
        const instance = new AirCanvasEngine(elements);
        setEngine(instance);
        let alive = true;
        void instance.init().then(() => {
          if (alive && !prefersReducedMotion()) void instance.startDemo();
        });
        return () => { alive = false; instance.dispose(); setEngine(null); };
      })
      .get();

  const keyboardShortcuts = () =>
    Maybe.of(engine)
      .map((eng) => {
        const actions = new Map<string, (e: KeyboardEvent) => void>([
          ['z', (e) => { if (e.shiftKey) eng.redo(); else eng.undo(); e.preventDefault(); }],
          ['y', () => eng.redo()],
          ['h', () => eng.toggleHand()],
        ]);

        const onKey = (ev: KeyboardEvent) =>
          Maybe.of(ev)
            .filter((e) => !(e.target as HTMLElement).closest('input, select, textarea, [contenteditable="true"]'))
            .filter((e) => !e.metaKey && !e.ctrlKey && !e.altKey)
            .map((e) => actions.get(e.key.toLowerCase()))
            .ifPresent((action) => action(ev));

        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
      })
      .get();

  useEffect(engineLifecycle, []);
  useEffect(keyboardShortcuts, [engine]);

  return (
    <div className="app">
      <Header />

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
      <Footer />
    </div>
  );
}

function Header() {
  return (
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
  );
}

function Footer() {
  return (
    <footer className="footer">
      <p>
        Air Canvas. No video, landmarks or drawing are uploaded anywhere.
      </p>
    </footer>
  );
}
