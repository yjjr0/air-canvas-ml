from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dataset import window_features
from gestures import LABELS, WINDOW

ENTER_THRESHOLD = 0.6
EXIT_THRESHOLD = 0.4
ENTER_FRAMES = {"pinch_draw": 2, "circle_clear": 4}
DEFAULT_ENTER_FRAMES = 3
EXIT_FRAMES = 2
MIN_FRAMES = 8


@dataclass
class GestureFilter:
    active: str | None = None
    candidate: str | None = None
    cand_count: int = 0
    exit_count: int = 0
    log: list = field(default_factory=list)

    def update(self, probs: dict[str, float]) -> tuple[str | None, str | None]:
        entered = exited = None
        if self.active is not None:
            self.exit_count = self.exit_count + 1 if probs.get(self.active, 0.0) < EXIT_THRESHOLD else 0
            if self.exit_count >= EXIT_FRAMES:
                exited, self.active, self.exit_count = self.active, None, 0
        top = max(probs, key=probs.get)
        if top != self.active and probs[top] >= ENTER_THRESHOLD:
            if top == self.candidate:
                self.cand_count += 1
            else:
                self.candidate, self.cand_count = top, 1
            if self.cand_count >= ENTER_FRAMES.get(top, DEFAULT_ENTER_FRAMES):
                if self.active is not None:
                    exited = self.active
                self.active, entered = top, top
                self.candidate, self.cand_count, self.exit_count = None, 0, 0
        else:
            self.candidate, self.cand_count = None, 0
        return entered, exited


def stream_windows(frames: np.ndarray, aspect: float) -> np.ndarray:
    """Every frame's trailing window, padded at the start like the browser buffer."""
    n = len(frames)
    feats = []
    for t in range(MIN_FRAMES - 1, n):
        idx = np.arange(t - WINDOW + 1, t + 1)
        start = max(0, t - WINDOW + 1)
        idx = np.maximum(idx, start)
        feats.append(window_features(frames[idx], aspect))
    return np.stack(feats)


def run_session(session, frames: np.ndarray, aspect: float) -> list[tuple[float, str, str]]:
    feats = stream_windows(frames, aspect)
    logits, _ = session.run(None, {"landmarks": feats})
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    filt = GestureFilter()
    events = []
    for i, p in enumerate(probs):
        t = (i + MIN_FRAMES - 1) / 30.0
        entered, exited = filt.update(dict(zip(LABELS, p.tolist())))
        if exited:
            events.append((t, "exit", exited))
        if entered:
            events.append((t, "enter", entered))
    return events


def main() -> None:
    import onnxruntime as ort

    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent.parent / "web/public/models/gesture_cnn.onnx")
    args = parser.parse_args()
    doc = json.loads(args.replay.read_text())
    frames = np.asarray(doc["frames"], dtype=np.float32).reshape(-1, 21, 3)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    for t, kind, label in run_session(session, frames, doc["aspect"]):
        if kind == "enter":
            print(f"{t:6.2f}s  {label}")


if __name__ == "__main__":
    main()
