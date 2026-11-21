from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gestures import CLIP_FRAMES, LABEL_TO_ID, LABELS, MOTION_LABELS, NUM_LANDMARKS, WINDOW

IDLE = LABEL_TO_ID["idle"]
MOTION_IDS = {LABEL_TO_ID[name] for name in MOTION_LABELS}
RECENT_FRAMES = 3 
MIN_CROP, MAX_CROP = 16, 24 


def to_isotropic(frames: np.ndarray, aspect: float) -> np.ndarray:
    p = np.array(frames, dtype=np.float64, copy=True)
    p[..., 0] *= aspect
    p[..., 2] *= aspect
    return p


def features_from_isotropic(p: np.ndarray) -> np.ndarray:
    """(B, T, 21, 3) isotropic landmarks -> (B, 126, T) float32 features."""
    palm = np.linalg.norm(p[:, :, 9] - p[:, :, 0], axis=-1)  
    scale = np.maximum(np.median(palm, axis=1), 1e-6)[:, None, None, None]
    pos = (p - p[:, :, :1]) / scale
    vel = np.zeros_like(p)
    vel[:, 1:] = (p[:, 1:] - p[:, :-1]) / scale
    b, t = p.shape[:2]
    feats = np.concatenate([pos.reshape(b, t, 63), vel.reshape(b, t, 63)], axis=2)
    return np.ascontiguousarray(feats.transpose(0, 2, 1)).astype(np.float32)


def window_features(window: np.ndarray, aspect: float) -> np.ndarray:
    """(T, 21, 3) MediaPipe-normalised landmarks -> (126, T) float32."""
    return features_from_isotropic(to_isotropic(np.asarray(window)[None], aspect))[0]


def load_recordings(data_dir: str | Path) -> list[dict]:
    """Load every ``*.json`` export from the web collector."""
    clips: list[dict] = []
    for path in sorted(Path(data_dir).glob("*.json")):
        doc = json.loads(path.read_text())
        if doc.get("format") != "air-canvas-clips":
            print(f"[data] skipping {path.name}: not an air-canvas-clips export")
            continue
        for i, clip in enumerate(doc.get("clips", [])):
            label = clip.get("label")
            if label not in LABEL_TO_ID:
                print(f"[data] {path.name}#{i}: unknown label {label!r} (custom gestures stay in the browser)")
                continue
            frames = np.asarray(clip["frames"], dtype=np.float32).reshape(-1, NUM_LANDMARKS, 3)
            if len(frames) < MAX_CROP:
                continue
            lid = LABEL_TO_ID[label]
            clips.append(
                {
                    "frames": frames,
                    "aspect": float(clip.get("aspect", doc.get("aspect", 4 / 3))),
                    "label": lid,
                    "frame_labels": np.full(len(frames), lid, dtype=np.int8),
                    "motion": None,
                    "kind": "single",
                    "group": str(clip.get("session") or doc.get("session") or path.stem),
                    "source": "real",
                }
            )
    return clips


def splice(a: dict, b: dict, rng: np.random.Generator, n: int = CLIP_FRAMES) -> dict:
    """Synthesise an A -> B transition from two real static clips.

    B is translated so its wrist continues from A's (no teleport = no fake
    velocity spike), then poses are cross-faded over 3-6 frames.
    """
    fa, fb = to_isotropic(a["frames"], a["aspect"]), to_isotropic(b["frames"], b["aspect"])
    n = min(n, len(fa), len(fb))
    cut = int(rng.integers(4, n - 6))
    fade = int(rng.integers(3, 7))
    sa = int(rng.integers(0, len(fa) - cut + 1))
    part_a = fa[sa : sa + cut]
    part_b = fb[: n - cut].copy()
    part_b[..., :2] += (part_a[-1, 0, :2] - part_b[0, 0, :2])[None, None]
    frames = np.concatenate([part_a, part_b])[:n]
    for k in range(fade):  
        w = (k + 1) / (fade + 1)
        t = cut + k
        if t < n:
            anchor = frames[t, 0].copy()
            pose_a = part_a[-1] - part_a[-1, 0]
            frames[t] = anchor + (1 - w) * pose_a + w * (frames[t] - frames[t, 0])
    aspect = a["aspect"]
    frames[..., 0] /= aspect
    frames[..., 2] /= aspect
    t = np.arange(n)
    labels = np.where(t < cut, a["label"], np.where(t < cut + fade, -1, b["label"])).astype(np.int8)
    return {"frames": frames.astype(np.float32), "aspect": aspect, "label": b["label"], "frame_labels": labels, "motion": None, "kind": "transition", "group": a["group"], "source": "real-splice"}


def motion_energy(clip: dict) -> np.ndarray:
    """Per-frame mean landmark speed (palm units / frame); cached on the clip."""
    if "_energy" not in clip:
        p = to_isotropic(clip["frames"], clip["aspect"])
        palm = np.median(np.linalg.norm(p[:, 9] - p[:, 0], axis=-1))
        speed = np.linalg.norm(np.diff(p, axis=0), axis=-1).mean(axis=-1) / max(palm, 1e-6)
        clip["_energy"] = np.concatenate([[0.0], speed])
    return clip["_energy"]


def choose_crop(clip: dict, length: int, rng: np.random.Generator | None) -> int:
    """Crop offset. Motion gestures: keep >= ~85% of the clip's motion in view."""
    n = len(clip["frames"])
    offsets = np.arange(n - length + 1)
    if clip["kind"] == "single" and clip["label"] in MOTION_IDS:
        energy = motion_energy(clip)
        cover = np.array([energy[o : o + length].sum() for o in offsets])
        good = offsets[cover >= 0.85 * cover.max()]
        return int(rng.choice(good)) if rng is not None else int(offsets[np.argmax(cover)])
    if rng is None:
        return int((n - length) // 2)
    return int(rng.integers(0, n - length + 1))


def crop_label(clip: dict, offset: int, length: int) -> int:
    """Label of a crop = what the hand is doing at its *end* (pen up/down responsiveness)."""
    if clip["kind"] == "single":
        return int(clip["label"])
    labels = clip["frame_labels"][offset : offset + length]
    tail = labels[-max(2, int(round(RECENT_FRAMES * length / WINDOW))) :]
    if tail[-1] >= 0 and np.all(tail == tail[-1]):
        return int(tail[-1])
    return IDLE


def resample_time(frames: np.ndarray, n_out: int) -> np.ndarray:
    n = len(frames)
    if n == n_out:
        return frames.copy()
    pos = np.linspace(0.0, n - 1, n_out)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (pos - lo)[:, None, None]
    return frames[lo] * (1 - w) + frames[hi] * w


def make_windows(
    clips: list[dict],
    rng: np.random.Generator | None,
    augment: bool,
    stress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """One window per clip. ``rng=None`` -> deterministic eval crops."""
    raw = np.empty((len(clips), WINDOW, NUM_LANDMARKS, 3))
    labels = np.empty(len(clips), dtype=np.int64)
    for i, clip in enumerate(clips):
        local_rng = rng if rng is not None else np.random.default_rng(i)
        if augment or stress:
            length = int(local_rng.integers(MIN_CROP, MAX_CROP + 1)) if augment else int(local_rng.choice([14, 28]))
            length = min(length, len(clip["frames"]))
        else:
            length = WINDOW
        if clip["kind"] == "transition":
            offset = int(local_rng.integers(0, len(clip["frames"]) - length + 1))
        else:
            offset = choose_crop(clip, length, rng if augment else None)
        labels[i] = crop_label(clip, offset, length)
        iso = to_isotropic(clip["frames"][offset : offset + length], clip["aspect"])
        raw[i] = resample_time(iso, WINDOW)
    if augment or stress:
        raw = augment_batch(raw, rng if rng is not None else np.random.default_rng(1234), strong=stress)
    return features_from_isotropic(raw), labels


def augment_batch(p: np.ndarray, rng: np.random.Generator, strong: bool = False) -> np.ndarray:
    """In-place-safe augmentations on isotropic windows (B, T, 21, 3)."""
    p = p.copy()
    b, t = p.shape[:2]
    max_rot = np.radians(30 if strong else 15)
    theta = rng.uniform(-max_rot, max_rot, b)
    c, s = np.cos(theta)[:, None, None], np.sin(theta)[:, None, None]
    origin = p[:, :1, :1, :2]
    xy = p[..., :2] - origin
    p[..., 0], p[..., 1] = c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]
    p[..., :2] += origin
    flip = rng.random(b) < 0.5
    p[flip, ..., 0] *= -1
    p[..., 2] *= rng.uniform(0.75, 1.25, (b, 1, 1))
    palm = np.median(np.linalg.norm(p[:, :, 9] - p[:, :, 0], axis=-1), axis=1)[:, None, None, None]
    sigma = rng.uniform(0.0, 0.03 if strong else 0.015, (b, 1, 1, 1)) * palm
    noise = rng.normal(0, 1, p.shape) * sigma
    noise[..., 2] *= 1.5
    p += noise
    if not strong:
        pad = rng.random(b) < 0.08
        for i in np.where(pad)[0]:
            k = int(rng.integers(1, 9))
            p[i, :k] = p[i, k]
        low = rng.random(b) < 0.15
        for i in np.where(low)[0]:
            p[i, 1:-1:2] = 0.5 * (p[i, 0:-2:2] + p[i, 2::2])
    return p


def class_names() -> list[str]:
    return list(LABELS)
