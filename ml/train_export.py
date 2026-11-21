from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from dataset import load_recordings, make_windows, splice, window_features
from gestures import CLIP_FRAMES, FEATURE_VERSION, FEATURES, FPS, LABEL_TO_ID, LABELS, MOTION_LABELS, STATIC_LABELS, WINDOW
from model import GestureNet1D, count_parameters
from synth import generate_dataset, generate_demo

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def synthetic_clips(args: argparse.Namespace) -> list[dict]:
    key = f"{args.synthetic_per_class}-{args.transitions}-{args.subjects}-{args.seed}-{hashlib.md5((ROOT / 'synth.py').read_bytes()).hexdigest()[:8]}"
    cache = ROOT / ".cache" / f"synth-{key}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    t0 = time.time()
    clips = generate_dataset(args.synthetic_per_class, args.transitions, args.subjects, args.seed)
    print(f"[data] generated {len(clips)} synthetic clips in {time.time() - t0:.1f}s")
    cache.parent.mkdir(exist_ok=True)
    cache.write_bytes(pickle.dumps(clips))
    return clips


def group_split(clips: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Hold out whole subjects/sessions so near-duplicate frames never straddle the split."""
    groups = sorted({c["group"] for c in clips})
    if len(groups) < 2:
        print("[data] only one recording session -> falling back to a per-clip split (expect optimistic val metrics)")
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(clips))
        cut = int(len(clips) * val_frac)
        return [clips[i] for i in idx[cut:]], [clips[i] for i in idx[:cut]]
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * val_frac)))
    val_groups = set(groups[:n_val])
    return [c for c in clips if c["group"] not in val_groups], [c for c in clips if c["group"] in val_groups]


def epoch_windows(synth: list[dict], real: list[dict], rng: np.random.Generator, real_weight: int) -> tuple[np.ndarray, np.ndarray]:
    clips = list(synth)
    if real:
        clips += real * real_weight
        statics = [c for c in real if LABELS[c["label"]] in STATIC_LABELS and len(c["frames"]) >= CLIP_FRAMES]
        if len({c["label"] for c in statics}) >= 2:
            for _ in range(len(real)):
                a, b = rng.choice(len(statics), 2, replace=False)
                if statics[a]["label"] != statics[b]["label"]:
                    clips.append(splice(statics[a], statics[b], rng))
    return make_windows(clips, rng, augment=True)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, embs = [], []
    for i in range(0, len(x), batch):
        lg, em = model(torch.from_numpy(x[i : i + batch]))
        logits.append(lg.numpy())
        embs.append(em.numpy())
    return np.concatenate(logits), np.concatenate(embs)


def metrics_for(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=range(len(LABELS)), zero_division=0)),
    }


def train(args: argparse.Namespace, synth_train, real_train, eval_sets) -> tuple[GestureNet1D, list[dict]]:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    model = GestureNet1D()
    x0, _ = epoch_windows(synth_train, real_train, np.random.default_rng(99), args.real_weight)
    mean = torch.from_numpy(x0.mean(axis=(0, 2)))
    std = torch.from_numpy(x0.std(axis=(0, 2)))
    model.set_normalization(mean, std)

    steps_per_epoch = int(np.ceil(len(x0) / args.batch_size))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch + 10, pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    select_on = "real_val" if "real_val" in eval_sets else "synthetic_val"
    best_state, best_score, history = None, -1.0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        x, y = epoch_windows(synth_train, real_train, rng, args.real_weight)
        model.train()
        perm = rng.permutation(len(y))
        total = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i : i + args.batch_size]
            if len(idx) < 2:
                continue
            logits, _ = model(torch.from_numpy(x[idx]))
            loss = loss_fn(logits, torch.from_numpy(y[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        row = {"epoch": epoch, "loss": total / len(y)}
        for name, (xv, yv) in eval_sets.items():
            row[name] = metrics_for(yv, predict(model, xv)[0].argmax(1))["macro_f1"]
        history.append(row)
        score = row[select_on]
        if score >= best_score:
            best_score, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"[train] epoch {epoch:02d} loss {row['loss']:.4f} " + " ".join(f"{k} {row[k]:.4f}" for k in eval_sets) + f" ({time.time() - t0:.1f}s)")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def export_onnx(model: GestureNet1D, path: Path, sample: np.ndarray) -> dict:
    import onnx
    import onnxruntime as ort

    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.from_numpy(sample[:1])
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["landmarks"],
        output_names=["logits", "embedding"],
        opset_version=18,
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
        dynamo=True,
        external_data=False,
        optimize=True,
        verbose=False,
    )
    proto = onnx.load(str(path))
    proto.graph.ClearField("value_info")
    proto.doc_string = "GestureNet1D: (batch,126,20) landmark window -> logits(7), L2-normalised embedding(32)"
    onnx.save(proto, str(path))
    onnx.checker.check_model(proto, full_check=True)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    report = {}
    for name, x in (("random", np.random.default_rng(0).normal(size=(16, FEATURES, WINDOW)).astype(np.float32)), ("windows", sample[:256])):
        with torch.no_grad():
            pt_logits, pt_emb = (t.numpy() for t in model(torch.from_numpy(x)))
        ort_logits, ort_emb = sess.run(None, {"landmarks": x})
        np.testing.assert_allclose(pt_logits, ort_logits, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(pt_emb, ort_emb, rtol=1e-3, atol=1e-5)
        report[name] = {"max_abs_logit_diff": float(np.abs(pt_logits - ort_logits).max()), "max_abs_emb_diff": float(np.abs(pt_emb - ort_emb).max())}
    return report


def benchmark(fp32_path: Path, x_eval: np.ndarray, y_eval: np.ndarray, out_dir: Path) -> dict:
    """FP32 vs INT8 (dynamic quantisation) latency, size and accuracy on CPU, 1 thread."""
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = out_dir / "gesture_cnn.int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8, op_types_to_quantize=["Conv", "MatMul", "Gemm"])
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    result = {}
    for name, path in (("fp32", fp32_path), ("int8", int8_path)):
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        one = x_eval[:1]
        for _ in range(200):
            sess.run(None, {"landmarks": one})
        times = []
        for i in range(2000):
            t0 = time.perf_counter()
            sess.run(None, {"landmarks": x_eval[i % len(x_eval) : i % len(x_eval) + 1]})
            times.append((time.perf_counter() - t0) * 1e6)
        logits = sess.run(None, {"landmarks": x_eval})[0]
        result[name] = {
            "size_kb": round(path.stat().st_size / 1024, 1),
            "latency_us_p50": round(float(np.percentile(times, 50)), 1),
            "latency_us_p95": round(float(np.percentile(times, 95)), 1),
            **metrics_for(y_eval, logits.argmax(1)),
        }
    return result


def class_prototypes(emb: np.ndarray, y: np.ndarray) -> dict:
    protos = {}
    for i, name in enumerate(LABELS):
        e = emb[y == i]
        if len(e) == 0:
            continue
        p = e.mean(axis=0)
        p /= np.linalg.norm(p) + 1e-9
        sims = e @ p
        protos[name] = {"vector": [round(float(v), 6) for v in p], "sim_p05": round(float(np.percentile(sims, 5)), 4)}
    return protos


def plot_reports(cm: np.ndarray, history: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.imshow(norm, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=40, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7, color="white" if norm[i, j] < 0.6 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Validation confusion matrix (held-out subjects)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    epochs = [h["epoch"] for h in history]
    ax1.plot(epochs, [h["loss"] for h in history], color="#888", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    for key, color in (("synthetic_val", "#d6249f"), ("stress", "#1f9fd6"), ("real_val", "#2ca02c")):
        if key in history[0]:
            ax2.plot(epochs, [h[key] for h in history], color=color, label=f"{key} macro-F1")
    ax2.set_ylabel("macro-F1")
    ax2.set_ylim(0, 1)
    fig.legend(loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)


def write_fixtures(path: Path, clips: list[dict], model: GestureNet1D, onnx_path: Path) -> None:
    """Raw windows + expected features/logits for the TypeScript parity test."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(5)
    cases = []
    by_label: dict[int, list[dict]] = {}
    for c in clips:
        by_label.setdefault(int(c["label"]), []).append(c)
    for lid in sorted(by_label):
        clip = by_label[lid][int(rng.integers(len(by_label[lid])))]
        raw = np.round(clip["frames"][:WINDOW].astype(np.float64), 6)
        feats = window_features(raw, clip["aspect"])
        logits, emb = sess.run(None, {"landmarks": feats[None]})
        cases.append(
            {
                "label": LABELS[lid],
                "aspect": clip["aspect"],
                "frames": [[round(float(v), 6) for v in frame.reshape(-1)] for frame in raw],
                "features": [round(float(v), 7) for v in feats.reshape(-1)],
                "logits": [round(float(v), 6) for v in logits[0]],
                "embedding": [round(float(v), 6) for v in emb[0]],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"feature_version": FEATURE_VERSION, "window": WINDOW, "cases": cases}))


def write_stream_fixture(path: Path, demo_path: Path, onnx_path: Path) -> None:
    """Per-frame probabilities of the demo replay + the events ml/stream.py's filter
    fires. The TypeScript GestureFilter must reproduce the same events."""
    import onnxruntime as ort

    from stream import EXIT_FRAMES, run_session, stream_windows

    demo = json.loads(demo_path.read_text())
    frames = np.asarray(demo["frames"], dtype=np.float32).reshape(-1, 21, 3)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logits = sess.run(None, {"landmarks": stream_windows(frames, demo["aspect"])})[0]
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    events = run_session(sess, frames, demo["aspect"])
    path.write_text(
        json.dumps(
            {
                "labels": list(LABELS),
                "exit_frames": EXIT_FRAMES,
                "probs": [[round(float(v), 6) for v in row] for row in probs],
                "events": [[round(t, 4), kind, label] for t, kind, label in events],
            }
        )
    )


def write_demo(path: Path) -> None:
    demo = generate_demo()
    demo["frames"] = [[round(float(v), 4) for v in f] for f in demo["frames"]]  
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(demo, separators=(",", ":")))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--synthetic-per-class", type=int, default=1000)
    p.add_argument("--transitions", type=int, default=2500)
    p.add_argument("--subjects", type=int, default=48)
    p.add_argument("--no-synthetic", action="store_true", help="train on real recordings only")
    p.add_argument("--real-weight", type=int, default=3, help="oversampling factor for real clips")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="write everything under this dir instead of web/ + ml/reports")
    args = p.parse_args()

    models_dir = (args.out or REPO / "web/public") / "models"
    demo_path = (args.out or REPO / "web/public") / "demo/demo_replay.json"
    fixtures_path = (args.out or REPO / "web/src/ml") / "__fixtures__/parity.json"
    reports_dir = args.out / "reports" if args.out else ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    synth = [] if args.no_synthetic else synthetic_clips(args)
    real = load_recordings(args.data_dir)
    print(f"[data] synthetic clips: {len(synth)}  real clips: {len(real)} from {len({c['group'] for c in real})} session(s)")
    if not synth and not real:
        raise SystemExit("no data: record clips in the web app (Collect tab) and put the JSON in ml/data/")

    synth_train, synth_val = group_split(synth, 0.2, args.seed) if synth else ([], [])
    real_train, real_val = group_split(real, 0.25, args.seed) if real else ([], [])

    eval_sets = {}
    if synth_val:
        eval_sets["synthetic_val"] = make_windows(synth_val, None, augment=False)
        eval_sets["stress"] = make_windows(synth_val, None, augment=False, stress=True)
    if real_val:
        eval_sets["real_val"] = make_windows(real_val, None, augment=False)

    model, history = train(args, synth_train, real_train, eval_sets)
    n_params = count_parameters(model)

    primary = "real_val" if "real_val" in eval_sets else "synthetic_val"
    xv, yv = eval_sets[primary]
    logits, emb = predict(model, xv)
    pred = logits.argmax(1)
    cm = confusion_matrix(yv, pred, labels=range(len(LABELS)))
    report = classification_report(yv, pred, labels=range(len(LABELS)), target_names=LABELS, output_dict=True, zero_division=0)
    print(classification_report(yv, pred, labels=range(len(LABELS)), target_names=LABELS, digits=3, zero_division=0))
    evals = {name: metrics_for(y, predict(model, x)[0].argmax(1)) for name, (x, y) in eval_sets.items()}

    onnx_path = models_dir / "gesture_cnn.onnx"
    parity = export_onnx(model, onnx_path, xv)
    print(f"[export] {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB) parity OK: {parity}")
    bench = benchmark(onnx_path, xv, yv, reports_dir)
    (reports_dir / "gesture_cnn.int8.onnx").unlink(missing_ok=True)
    print(f"[bench] {json.dumps(bench)}")

    train_eval_clips = synth_train + real_train
    xt, yt = make_windows(train_eval_clips, None, augment=False)
    _, emb_train = predict(model, xt)
    protos = class_prototypes(emb_train, yt)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {
        "model": "gesture_cnn.onnx",
        "labels": list(LABELS),
        "static_labels": list(STATIC_LABELS),
        "motion_labels": list(MOTION_LABELS),
        "input": {"name": "landmarks", "shape": [1, FEATURES, WINDOW]},
        "outputs": ["logits", "embedding"],
        "fps": FPS,
        "window": WINDOW,
        "feature_version": FEATURE_VERSION,
        "embedding_dim": int(emb.shape[1]),
        "params": n_params,
        "onnx_kb": round(onnx_path.stat().st_size / 1024, 1),
        "prototypes": protos,
        "trained_at": stamp,
        "data": {"synthetic_clips": len(synth), "real_clips": len(real), "real_sessions": len({c["group"] for c in real})},
        "metrics": {name: {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()} for name, m in evals.items()},
    }
    (models_dir / "gesture_meta.json").write_text(json.dumps(meta, indent=1))
    write_fixtures(fixtures_path, synth_val or real_val, model, onnx_path)
    write_demo(demo_path)
    write_stream_fixture(fixtures_path.parent / "stream.json", demo_path, onnx_path)

    reports_dir.mkdir(parents=True, exist_ok=True)
    plot_reports(cm, history, reports_dir)
    (reports_dir / "metrics.json").write_text(
        json.dumps(
            {
                "trained_at": stamp,
                "primary_eval": primary,
                "params": n_params,
                "evals": evals,
                "per_class": {k: {m: round(v, 4) for m, v in report[k].items()} for k in LABELS},
                "confusion_matrix": cm.tolist(),
                "onnx_parity": parity,
                "benchmark_cpu_1_thread": bench,
                "history": history,
                "args": {k: str(v) for k, v in vars(args).items()},
            },
            indent=1,
        )
    )
    print(f"[done] meta, fixtures, demo replay and reports written ({primary}: {evals[primary]})")


if __name__ == "__main__":
    main()
