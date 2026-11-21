import { REPO_URL } from '../config';
import type { GestureMeta } from '../ml/meta';

export function HowItWorks({ meta }: { meta: GestureMeta | null }) {
  const val = meta?.metrics.real_val ?? meta?.metrics.synthetic_val;
  const onReal = Boolean(meta?.metrics.real_val);
  return (
    <section className="how" aria-labelledby="how-title">
      <h2 id="how-title">How it works</h2>
      <ol className="pipeline">
        <li>
          <h3>Hand Motion Tracking</h3>
          <p>MediaPipe Hand Landmarker tracks 21 points on your hand.</p>
        </li>
        <li>
          <h3>Fixed Frame Rate</h3>
          <p>All frames are resampled to exactly 30 FPS, the rate the model was trained on.</p>
        </li>
        <li>
          <h3>Classify</h3>
          <p>
            A {meta ? `${(meta.params / 1000).toFixed(0)}K-parameter` : 'small'} 1D convolutional network, exported to ONNX and run by ONNX Runtime Web,
            outputs a probability for each gesture and a 32-number embedding used by Teach.
          </p>
        </li>
      </ol>
      <div className="model-card">
        <h3>About the model</h3>
        <p>
          {meta ? (
            <>
              Trained on {meta.data.synthetic_clips.toLocaleString()} clips from a procedural hand generator
              {meta.data.real_clips > 0 ? ` and ${meta.data.real_clips.toLocaleString()} recorded clips` : ''}.
              {val && (
                <>
                  {' '}
                  Macro-F1 {val.macro_f1.toFixed(3)} on {onReal ? 'held-out recording sessions' : 'held-out synthetic hands'}.
                </>
              )}
            </>
          ) : (
            'Loading model details…'
          )}
        </p>
        <p>
          Training code, the synthetic data generator, evaluation and the export parity checks are in the{' '}
          <a href={REPO_URL}>repository</a>.
        </p>
      </div>
    </section>
  );
}
