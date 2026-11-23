from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gestures import CLIP_FRAMES, FPS, LABEL_TO_ID, LABELS, STATIC_LABELS


MCP_BASE = np.array(
    [
        [0.27, 0.95, 0.02],  
        [0.00, 1.00, 0.00],  
        [-0.22, 0.93, 0.00],  
        [-0.44, 0.81, 0.03],  
    ]
)
SPLAY_REST = np.radians([7.0, 0.0, -7.0, -15.0])  
SPLAY_SPREAD = np.radians([9.0, 1.0, -7.0, -15.0])  
PHALANX_BASE = np.array(
    [
        [0.40, 0.23, 0.19], 
        [0.45, 0.27, 0.20],
        [0.42, 0.26, 0.20],
        [0.33, 0.19, 0.18],
    ]
)
THUMB_CMC_BASE = np.array([0.22, 0.18, 0.03])
THUMB_LEN_BASE = np.array([0.40, 0.30, 0.26])  
FINGER_JOINTS = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)  

ASPECTS = (4 / 3, 16 / 9)


@dataclass(frozen=True)
class Subject:
    """A virtual person: bone proportions + tracker noise level."""

    mcp: np.ndarray
    phalanx: np.ndarray
    thumb_cmc: np.ndarray
    thumb_len: np.ndarray
    flex_bias: float  
    jitter: float 


@dataclass
class PoseSpec:
    curl: np.ndarray 
    spread: float 
    extra_splay: np.ndarray  
    cup: float  
    thumb: dict  
    curl_jitter: float = 0.04 


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def smooth_noise(rng: np.random.Generator, n: int, dims: int, sigma: float, scale: float) -> np.ndarray:
    """Gaussian-filtered white noise with (theoretical) per-sample std ``scale``."""
    if scale == 0:
        return np.zeros((n, dims))
    sigma = max(sigma, 0.5)
    pad = int(3 * sigma) + 1
    kernel = np.exp(-0.5 * (np.arange(-pad, pad + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    white = rng.normal(0.0, 1.0, (n + 2 * pad, dims))
    out = np.stack([np.convolve(white[:, d], kernel, mode="valid") for d in range(dims)], axis=-1)
    return out / np.sqrt(np.sum(kernel**2)) * scale


def min_jerk(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def smoothstep(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(tau, 0.0, 1.0)
    return tau * tau * (3 - 2 * tau)


def rotation_stack(roll: np.ndarray, yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """(N,3,3) rotations taking the local hand frame into the camera frame.

    Base mapping: local +X -> image +x, +Y (fingers) -> image up (-y),
    +Z (palm normal) -> toward the camera (-z). Then pitch, yaw, roll.
    """
    n = roll.shape[0]
    cr, sr = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.zeros((n, 3, 3))
    rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1], rz[:, 2, 2] = cr, -sr, sr, cr, 1
    ry = np.zeros((n, 3, 3))
    ry[:, 0, 0], ry[:, 0, 2], ry[:, 1, 1], ry[:, 2, 0], ry[:, 2, 2] = cy, sy, 1, -sy, cy
    rx = np.zeros((n, 3, 3))
    rx[:, 0, 0], rx[:, 1, 1], rx[:, 1, 2], rx[:, 2, 1], rx[:, 2, 2] = 1, cp, -sp, sp, cp
    base = np.diag([1.0, -1.0, -1.0])
    return rz @ ry @ rx @ base


def sample_subject(rng: np.random.Generator) -> Subject:
    mcp = MCP_BASE.copy()
    mcp[:, 0] *= rng.uniform(0.88, 1.12)  
    mcp[:, 1] += rng.normal(0.0, 0.02, 4)
    phalanx = PHALANX_BASE * rng.uniform(0.9, 1.12) * rng.uniform(0.95, 1.05, (4, 3))
    thumb_len = THUMB_LEN_BASE * rng.uniform(0.92, 1.1) * rng.uniform(0.96, 1.04, 3)
    thumb_cmc = THUMB_CMC_BASE + rng.normal(0.0, 0.02, 3)
    return Subject(
        mcp=mcp,
        phalanx=phalanx,
        thumb_cmc=thumb_cmc,
        thumb_len=thumb_len,
        flex_bias=float(rng.uniform(-0.3, 0.3)),
        jitter=float(rng.uniform(0.006, 0.02)),
    )


def fingers_fk(subject: Subject, curl: np.ndarray, spread: np.ndarray, extra_splay: np.ndarray, cup: np.ndarray) -> np.ndarray:
    """Forward kinematics for the four fingers. Returns (N, 21, 3); thumb filled later."""
    n = curl.shape[0]
    pts = np.zeros((n, 21, 3))
    c = np.clip(curl, -0.2, 1.05)
    bias = subject.flex_bias
    mcp_flex = np.radians(82.0) * c * (1 + 0.3 * bias)
    pip_flex = np.clip(np.radians(102.0) * c * (1 - 0.3 * bias), 0.0, None)
    dip_flex = 0.68 * pip_flex
    splay = (0.6 * SPLAY_REST + spread[:, None] * SPLAY_SPREAD + extra_splay) * (1 - 0.6 * np.clip(c, 0, 1))
    d = np.stack([np.sin(splay), np.cos(splay), np.zeros_like(splay)], axis=-1) 
    normal = np.array([0.0, 0.0, 1.0])
    mcp = np.broadcast_to(subject.mcp, (n, 4, 3)).copy()
    mcp[..., 2] += cup[:, None] * np.array([0.05, 0.0, 0.03, 0.08])
    joints = [mcp]
    phi = np.zeros_like(c)
    for k, flex in enumerate((mcp_flex, pip_flex, dip_flex)):
        phi = phi + flex
        u = np.cos(phi)[..., None] * d + np.sin(phi)[..., None] * normal
        joints.append(joints[-1] + subject.phalanx[None, :, k, None] * u)
    for f, ids in enumerate(FINGER_JOINTS):
        for j, lid in enumerate(ids):
            pts[:, lid] = joints[j][:, f]
    pts[:, 1] = subject.thumb_cmc + cup[:, None] * np.array([0.0, 0.0, 0.04])
    return pts


def thumb_target(spec: dict, pts: np.ndarray) -> np.ndarray:
    """Where the thumb tip should be, given the already-posed fingers. (N, 3)."""
    n = pts.shape[0]
    cmc = pts[:, 1]
    mode = spec["mode"]
    offset = spec.get("offset_track", np.zeros((n, 3)))
    if mode in ("open", "relaxed"):
        abd, fwd, reach = spec["abd"], spec["fwd"], spec["reach"]
        direction = _unit(np.stack([np.sin(abd), np.cos(abd), fwd], axis=-1))
        return cmc + direction * reach[:, None] + offset
    if mode == "tuck":
        a, b = spec["anchors"]
        return 0.5 * (pts[:, a] + pts[:, b]) + spec["offset"] + offset
    if mode == "side":
        return pts[:, spec["anchor"]] + spec["offset"] + offset
    if mode == "pinch":
        return pts[:, 8] + spec["offset"] + offset
    raise ValueError(f"unknown thumb mode {mode!r}")


def thumb_ik(cmc: np.ndarray, target: np.ndarray, lengths: np.ndarray, iters: int = 14) -> np.ndarray:
    """Vectorised FABRIK for the 3-bone thumb chain. Returns (N, 3, 3): MCP, IP, TIP."""
    n = target.shape[0]
    p = np.zeros((n, 4, 3))
    p[:, 0] = cmc
    p[:, 1] = p[:, 0] + lengths[0] * _unit([0.62, 0.62, 0.48])
    p[:, 2] = p[:, 1] + lengths[1] * _unit([0.30, 0.78, 0.55])
    p[:, 3] = p[:, 2] + lengths[2] * _unit([0.05, 0.80, 0.60])
    for _ in range(iters):
        p[:, 3] = target
        for i in (2, 1, 0):
            p[:, i] = p[:, i + 1] + lengths[i] * _unit(p[:, i] - p[:, i + 1])
        p[:, 0] = cmc
        for i in (0, 1, 2):
            p[:, i + 1] = p[:, i] + lengths[i] * _unit(p[:, i + 1] - p[:, i])
    return p[:, 1:]


def _thumb_open(rng: np.random.Generator) -> dict:
    return {"mode": "open", "abd": np.radians(rng.uniform(25, 65)), "fwd": rng.uniform(0.0, 0.35), "reach": rng.uniform(0.84, 0.95)}


def _thumb_relaxed(rng: np.random.Generator) -> dict:
    return {"mode": "relaxed", "abd": np.radians(rng.uniform(18, 42)), "fwd": rng.uniform(0.45, 0.9), "reach": rng.uniform(0.76, 0.88)}


def _thumb_tuck(rng: np.random.Generator, anchors: tuple[int, int]) -> dict:
    return {"mode": "tuck", "anchors": anchors, "offset": np.array([-0.06, -0.14, 0.12]) + rng.normal(0, 0.035, 3)}


def pose_open(rng: np.random.Generator, subject: Subject) -> PoseSpec:
    return PoseSpec(
        curl=rng.uniform(-0.12, 0.12, 4),
        spread=rng.uniform(0.4, 1.0),
        extra_splay=rng.normal(0, np.radians(2.5), 4),
        cup=rng.uniform(0.0, 0.5),
        thumb=_thumb_open(rng),
    )


def pose_fist(rng: np.random.Generator, subject: Subject) -> PoseSpec:
    if rng.random() < 0.7:
        thumb = _thumb_tuck(rng, (6, 10))
    else:  
        thumb = {"mode": "side", "anchor": 6, "offset": np.array([0.13, -0.12, 0.06]) + rng.normal(0, 0.03, 3)}
    return PoseSpec(
        curl=rng.uniform(0.82, 1.0, 4),
        spread=rng.uniform(0.0, 0.2),
        extra_splay=rng.normal(0, np.radians(2), 4),
        cup=rng.uniform(0.3, 1.0),
        thumb=thumb,
        curl_jitter=0.025,
    )


def pose_pinch(rng: np.random.Generator, subject: Subject) -> PoseSpec:
    variant = rng.choice(["ok", "relaxed", "closed"], p=[0.35, 0.4, 0.25])
    others = {"ok": (-0.05, 0.3), "relaxed": (0.25, 0.65), "closed": (0.65, 1.0)}[variant]
    curl = np.concatenate([[rng.uniform(0.5, 0.8)], rng.uniform(*others, 3)])
    spec = PoseSpec(
        curl=curl,
        spread=rng.uniform(0.1, 0.6),
        extra_splay=rng.normal(0, np.radians(2.5), 4),
        cup=rng.uniform(0.0, 0.6),
        thumb={"mode": "pinch", "offset": np.array([0.03, -0.03, 0.03]) + rng.normal(0, 0.015, 3)},
        curl_jitter=0.025,
    )
    reach = subject.thumb_len.sum() * 0.98
    for _ in range(8):
        pts = fingers_fk(subject, spec.curl[None], np.array([spec.spread]), spec.extra_splay[None], np.array([spec.cup]))
        if np.linalg.norm(pts[0, 8] + spec.thumb["offset"] - pts[0, 1]) <= reach:
            break
        spec.curl[0] = min(spec.curl[0] + 0.05, 0.98)
    return spec


def pose_peace(rng: np.random.Generator, subject: Subject) -> PoseSpec:
    v = np.radians(rng.uniform(5, 16))
    return PoseSpec(
        curl=np.array([rng.uniform(-0.1, 0.12), rng.uniform(-0.1, 0.12), rng.uniform(0.78, 1.0), rng.uniform(0.78, 1.0)]),
        spread=rng.uniform(0.6, 1.0),
        extra_splay=np.array([v, -v * rng.uniform(0.6, 1.2), 0.0, 0.0]) + rng.normal(0, np.radians(2), 4),
        cup=rng.uniform(0.2, 0.8),
        thumb={"mode": "tuck", "anchors": (14, 18), "offset": np.array([0.08, -0.12, 0.12]) + rng.normal(0, 0.035, 3)},
        curl_jitter=0.03,
    )


def pose_point(rng: np.random.Generator, subject: Subject) -> PoseSpec:
    return PoseSpec(
        curl=np.array([rng.uniform(-0.1, 0.15), rng.uniform(0.75, 1.0), rng.uniform(0.75, 1.0), rng.uniform(0.75, 1.0)]),
        spread=rng.uniform(0.0, 0.35),
        extra_splay=rng.normal(0, np.radians(2.5), 4),
        cup=rng.uniform(0.2, 0.9),
        thumb=_thumb_tuck(rng, (10, 14)),
        curl_jitter=0.03,
    )


def pose_relaxed(rng: np.random.Generator, subject: Subject) -> PoseSpec:
    base = rng.uniform(0.15, 0.55)
    cascade = np.array([0.0, 0.05, 0.1, 0.15]) * rng.uniform(0.0, 1.5)
    return PoseSpec(
        curl=np.clip(base + cascade + rng.normal(0, 0.06, 4), 0.13, 0.75),
        spread=rng.uniform(0.0, 0.6),
        extra_splay=rng.normal(0, np.radians(3), 4),
        cup=rng.uniform(0.2, 1.0),
        thumb=_thumb_relaxed(rng),
        curl_jitter=0.05,
    )


def pose_three(rng: np.random.Generator, subject: Subject) -> PoseSpec:
    return PoseSpec(
        curl=np.array([rng.uniform(-0.1, 0.12), rng.uniform(-0.1, 0.12), rng.uniform(-0.1, 0.15), rng.uniform(0.75, 1.0)]),
        spread=rng.uniform(0.3, 0.9),
        extra_splay=rng.normal(0, np.radians(2.5), 4),
        cup=rng.uniform(0.0, 0.6),
        thumb=_thumb_tuck(rng, (14, 18)),
    )


def pose_thumb_out_fist(rng: np.random.Generator, subject: Subject) -> PoseSpec:
    """Curled fingers with the thumb sticking out (thumbs-up / hitch-hiker)."""
    return PoseSpec(
        curl=rng.uniform(0.8, 1.0, 4),
        spread=rng.uniform(0.0, 0.2),
        extra_splay=rng.normal(0, np.radians(2), 4),
        cup=rng.uniform(0.3, 1.0),
        thumb={"mode": "open", "abd": np.radians(rng.uniform(5, 40)), "fwd": rng.uniform(0.1, 0.5), "reach": rng.uniform(0.86, 0.95)},
    )


def pose_random(rng: np.random.Generator, subject: Subject) -> PoseSpec:
    """Uniformly random hand, rejected if it looks like one of the named gestures."""
    for _ in range(50):
        curl = rng.uniform(-0.1, 1.0, 4)
        spec = PoseSpec(
            curl=curl,
            spread=rng.uniform(0, 1),
            extra_splay=rng.normal(0, np.radians(4), 4),
            cup=rng.uniform(0, 1),
            thumb=_thumb_relaxed(rng) if rng.random() < 0.5 else _thumb_open(rng),
        )
        open_like = np.all(curl < 0.25)
        fist_like = np.all(curl > 0.7)
        peace_like = curl[0] < 0.3 and curl[1] < 0.3 and curl[2] > 0.6 and curl[3] > 0.6
        if not (open_like or fist_like or peace_like):
            return spec
    return pose_relaxed(rng, subject)


def thumb_index_gap(spec: PoseSpec, subject: Subject, rng: np.random.Generator) -> float:
    """Thumb-tip to index-tip distance (palm units) for the spec's base pose."""
    pts = local_landmarks(subject, [spec], np.zeros(1), rng)[0]
    return float(np.linalg.norm(pts[4] - pts[8]))


def not_a_pinch(sampler, min_gap: float = 0.32):
    """Wrap a pose sampler so the thumb never ends up touching the index tip.

    Keeps a margin between "idle" hands and the pinch class, so label noise does
    not teach the model that a near-pinch is idle (or vice versa)."""

    def wrapped(rng: np.random.Generator, subject: Subject) -> PoseSpec:
        spec = sampler(rng, subject)
        for _ in range(20):
            if thumb_index_gap(spec, subject, rng) >= min_gap:
                return spec
            spec = sampler(rng, subject)
        spec.thumb = _thumb_open(rng)
        return spec

    wrapped.__name__ = f"not_a_pinch({sampler.__name__})"
    return wrapped


def pose_idle(rng: np.random.Generator, subject: Subject) -> PoseSpec:
    """Everything that should *not* trigger an action."""
    sampler = rng.choice(
        [pose_relaxed, pose_point, pose_three, pose_thumb_out_fist, pose_random],
        p=[0.42, 0.2, 0.1, 0.08, 0.2],
    )
    return not_a_pinch(sampler)(rng, subject)


STATIC_POSES = {
    "idle": pose_idle,
    "pinch_draw": pose_pinch,
    "open_palm_hover": pose_open,
    "fist_erase": pose_fist,
    "peace_color": pose_peace,
}


def _thumb_track(spec: dict, rng: np.random.Generator, n: int, amp: float) -> dict:
    out = dict(spec)
    out["offset_track"] = smooth_noise(rng, n, 3, sigma=6, scale=amp)
    for key in ("abd", "fwd", "reach"):
        if key in spec:
            jitter = {"abd": np.radians(4), "fwd": 0.05, "reach": 0.015}[key]
            out[key] = spec[key] + smooth_noise(rng, n, 1, sigma=7, scale=jitter)[:, 0]
    return out


def local_landmarks(subject: Subject, specs: list[PoseSpec], weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blend 1 or 2 pose specs over time and solve the hand. ``weights``: (N,) weight of specs[1]."""
    n = weights.shape[0]
    tracks = []
    for spec in specs:
        tracks.append(
            {
                "curl": spec.curl[None] + smooth_noise(rng, n, 4, sigma=6, scale=spec.curl_jitter),
                "spread": spec.spread + smooth_noise(rng, n, 1, sigma=8, scale=0.05)[:, 0],
                "splay": spec.extra_splay[None] + smooth_noise(rng, n, 4, sigma=8, scale=np.radians(1.5)),
                "cup": spec.cup + smooth_noise(rng, n, 1, sigma=8, scale=0.05)[:, 0],
                "thumb": _thumb_track(spec.thumb, rng, n, amp=0.012 if spec.thumb["mode"] == "pinch" else 0.03),
            }
        )
    if len(tracks) == 1:
        tracks.append(tracks[0])
    w = weights
    curl = (1 - w[:, None]) * tracks[0]["curl"] + w[:, None] * tracks[1]["curl"]
    spread = np.clip((1 - w) * tracks[0]["spread"] + w * tracks[1]["spread"], 0.0, 1.1)
    splay = (1 - w[:, None]) * tracks[0]["splay"] + w[:, None] * tracks[1]["splay"]
    cup = np.clip((1 - w) * tracks[0]["cup"] + w * tracks[1]["cup"], 0.0, 1.0)
    pts = fingers_fk(subject, curl, spread, splay, cup)
    target = (1 - w[:, None]) * thumb_target(tracks[0]["thumb"], pts) + w[:, None] * thumb_target(tracks[1]["thumb"], pts)
    pts[:, 2:5] = thumb_ik(pts[:, 1], target, subject.thumb_len)
    return pts


def traj_drift(rng: np.random.Generator, n: int, amp: float) -> np.ndarray:
    return smooth_noise(rng, n, 2, sigma=rng.uniform(5, 10), scale=amp)


def traj_hover(rng: np.random.Generator, n: int, max_speed: float = 4.5, fast_speed: float = 9.0) -> np.ndarray:
    """Cursor-style repositioning: drift plus an optional directed move.

    Hard negatives for the motion gestures: moves that are *not* near-horizontal
    may be fast (up to ``fast_speed`` palm/s) - fast is not a swipe unless it is
    sideways - and some moves follow arcs of up to 180 deg - curved is not a
    circle unless it closes the loop. Near-horizontal moves stay below
    ``max_speed`` so the swipe boundary remains learnable.
    """
    path = traj_drift(rng, n, rng.uniform(0.04, 0.3))
    if rng.random() < 0.75:
        dur = rng.uniform(10, 30)
        ang = rng.uniform(0, 2 * np.pi)
        limit = max_speed if abs(np.sin(ang)) < 0.5 else fast_speed
        dist = rng.uniform(0.3, max(0.31, limit * (dur / FPS) / 1.875))
        t0 = rng.uniform(-dur * 0.5, n - dur * 0.5)
        s = min_jerk((np.arange(n) - t0) / dur)
        if rng.random() < 0.3: 
            sweep = np.radians(rng.uniform(40, 180)) * rng.choice([-1, 1])
            r = dist / max(abs(sweep), 1e-3)
            phi = ang + sweep * s
            arc = r * np.stack([np.sin(phi) - np.sin(ang), -(np.cos(phi) - np.cos(ang))], axis=-1) * np.sign(sweep)
            path += arc
        else:
            path += dist * s[:, None] * np.array([np.cos(ang), np.sin(ang)])
    return path


def traj_draw(rng: np.random.Generator, n: int) -> np.ndarray:
    """Pinch strokes: lines, arcs/full circles, scribbles, free curves (hard negatives)."""
    t = np.arange(n) / FPS
    kind = rng.choice(["still", "line", "arc", "wave", "curve"], p=[0.15, 0.25, 0.25, 0.15, 0.2])
    speed = rng.uniform(0.5, 4.0)
    path = traj_drift(rng, n, rng.uniform(0.02, 0.12))
    if kind == "line":
        ang = rng.uniform(0, 2 * np.pi)
        path += np.outer(rng.uniform(0.5, 5.0) * t, [np.cos(ang), np.sin(ang)])
    elif kind == "arc":
        r = rng.uniform(0.3, 1.3)
        omega = speed / r * rng.choice([-1, 1])
        phi0 = rng.uniform(0, 2 * np.pi)
        phi = phi0 + omega * t
        path += r * np.stack([np.cos(phi) - np.cos(phi0), np.sin(phi) - np.sin(phi0)], axis=-1)
    elif kind == "wave":
        ang = rng.uniform(0, 2 * np.pi)
        along = np.array([np.cos(ang), np.sin(ang)])
        perp = np.array([-along[1], along[0]])
        amp, freq = rng.uniform(0.1, 0.5), rng.uniform(1.0, 3.0)
        path += np.outer(speed * 0.6 * t, along) + np.outer(amp * np.sin(2 * np.pi * freq * t), perp)
    elif kind == "curve":
        path += smooth_noise(rng, n, 2, sigma=rng.uniform(4, 8), scale=rng.uniform(0.3, 1.0))
    return path


def traj_scrub(rng: np.random.Generator, n: int) -> np.ndarray:
    """Eraser motion: back-and-forth scrubbing or plain hovering."""
    if rng.random() < 0.5:
        return traj_hover(rng, n)
    t = np.arange(n) / FPS
    ang = rng.uniform(0, 2 * np.pi)
    amp, freq = rng.uniform(0.2, 1.0), rng.uniform(0.8, 2.2)
    osc = amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    return np.outer(osc, [np.cos(ang), np.sin(ang)]) + traj_drift(rng, n, 0.1)


def traj_swipe(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast horizontal flick with a minimum-jerk profile. Returns path, orientation delta, motion mask."""
    amp = rng.uniform(2.0, 4.5)
    dur = rng.uniform(0.25, 0.5) * FPS
    ang = np.radians(rng.uniform(-15, 15)) + (np.pi if rng.random() < 0.5 else 0.0)
    t0 = rng.uniform(2, max(2.5, n - dur - 2))
    tau = (np.arange(n) - t0) / dur
    s = min_jerk(tau)
    direction = np.array([np.cos(ang), np.sin(ang)])
    perp = np.array([-direction[1], direction[0]])
    sag = rng.uniform(-0.25, 0.25) * amp * np.sin(np.pi * s)
    path = np.outer((s - 0.5) * amp, direction) + np.outer(sag, perp) + traj_drift(rng, n, 0.05)
    orient = np.zeros((n, 3))
    orient[:, 0] = np.radians(rng.uniform(-12, 12)) * np.sin(np.pi * s)
    orient[:, 1] = np.radians(rng.uniform(-15, 15)) * (s - 0.5)
    mask = (tau > 0) & (tau < 1)
    return path, orient, mask


def traj_circle(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One quick loop, either moving the whole hand or pivoting at the wrist."""
    dur = rng.uniform(0.45, 0.85) * FPS
    t0 = rng.uniform(1, max(1.5, n - dur - 1))
    tau = np.clip((np.arange(n) - t0) / dur, 0, 1)
    grid = np.linspace(0, 1, 200)
    speed = np.minimum(1.0, np.minimum(grid / 0.15, (1 - grid) / 0.15)) + 1e-3
    prog = np.cumsum(speed)
    prog = (prog - prog[0]) / (prog[-1] - prog[0])
    p = np.interp(tau, grid, prog)
    sweep = np.radians(rng.uniform(300, 420)) * rng.choice([-1, 1])
    phi0 = rng.uniform(0, 2 * np.pi)
    phi = phi0 + sweep * p
    mask = (np.arange(n) > t0) & (np.arange(n) < t0 + dur)
    orient = np.zeros((n, 3))
    path = traj_drift(rng, n, 0.04)
    if rng.random() < 0.7:  
        r, k = rng.uniform(0.4, 1.3), rng.uniform(0.7, 1.3)
        path += r * np.stack([np.cos(phi) - np.cos(phi0), k * (np.sin(phi) - np.sin(phi0))], axis=-1)
    else:  
        a = np.radians(rng.uniform(10, 22))
        env = np.sin(np.pi * p) ** 0.5
        orient[:, 1] = a * np.cos(phi) * env
        orient[:, 2] = a * np.sin(phi) * env
        path += 0.25 * np.stack([np.cos(phi) - np.cos(phi0), np.sin(phi) - np.sin(phi0)], axis=-1)
    return path, orient, mask


@dataclass
class View:
    roll: float
    yaw: float
    pitch: float
    left: bool
    scale: float  
    aspect: float
    center: np.ndarray  
    zscale: float  


def sample_view(rng: np.random.Generator, label: str) -> View:
    back_prob = {"fist_erase": 0.3, "idle": 0.2, "pinch_draw": 0.08}.get(label, 0.12)
    yaw = np.radians(rng.uniform(-55, 55)) + (np.pi if rng.random() < back_prob else 0.0)
    aspect = float(rng.choice(ASPECTS))
    return View(
        roll=float(np.radians(np.clip(rng.normal(0, 20), -60, 60))),
        yaw=float(yaw),
        pitch=float(np.radians(rng.uniform(-35, 30))),
        left=bool(rng.random() < 0.5),
        scale=float(rng.uniform(0.1, 0.28)),
        aspect=aspect,
        center=np.array([rng.uniform(0.3, 0.7) * aspect, rng.uniform(0.5, 0.85)]),
        zscale=float(rng.uniform(0.6, 1.1)),
    )


def project(
    local: np.ndarray,
    view: View,
    path: np.ndarray,
    orient: np.ndarray,
    rng: np.random.Generator,
    jitter: float,
    orient_noise: float = np.radians(3),
) -> np.ndarray:
    """Local landmarks (N,21,3) -> MediaPipe-normalised image coords (N,21,3)."""
    n = local.shape[0]
    pts = local.copy()
    if view.left:
        pts[..., 0] *= -1.0
    wobble = smooth_noise(rng, n, 3, sigma=8, scale=orient_noise)
    roll = view.roll + orient[:, 0] + wobble[:, 0]
    yaw = view.yaw + orient[:, 1] + wobble[:, 1]
    pitch = view.pitch + orient[:, 2] + wobble[:, 2]
    rot = rotation_stack(roll, yaw, pitch)
    scale = view.scale * (1 + smooth_noise(rng, n, 1, sigma=10, scale=0.04)[:, 0])
    cam = np.einsum("nij,nkj->nki", rot, pts) * scale[:, None, None]
    noise = rng.normal(0, 1, cam.shape) * jitter + smooth_noise(rng, n, 63, sigma=3, scale=jitter * 0.6).reshape(n, 21, 3)
    noise[..., 2] *= 1.8
    cam += noise * scale[:, None, None]
    wrist = view.center[None] + path * view.scale
    out = np.empty_like(cam)
    out[..., 0] = (wrist[:, None, 0] + cam[..., 0]) / view.aspect
    out[..., 1] = wrist[:, None, 1] + cam[..., 1]
    out[..., 2] = cam[..., 2] * view.zscale / view.aspect
    return out


def generate_clip(label: str, rng: np.random.Generator, subject: Subject, n: int = CLIP_FRAMES) -> dict:
    """One single-gesture clip of ``n`` frames at FPS."""
    view = sample_view(rng, label)
    orient = np.zeros((n, 3))
    motion = np.zeros(n, dtype=bool)
    if label in STATIC_POSES:
        spec = STATIC_POSES[label](rng, subject)
        if label == "pinch_draw":
            path = traj_draw(rng, n)
        elif label == "open_palm_hover":
            path = traj_hover(rng, n)
        elif label == "fist_erase":
            path = traj_scrub(rng, n)
        elif label == "idle" and rng.random() < 0.4:
            path = traj_hover(rng, n)  
        else:
            path = traj_drift(rng, n, rng.uniform(0.04, 0.4))
    elif label == "swipe_undo":
        spec = pose_open(rng, subject) if rng.random() < 0.7 else not_a_pinch(pose_relaxed)(rng, subject)
        path, orient, motion = traj_swipe(rng, n)
    elif label == "circle_clear":
        spec = pose_point(rng, subject) if rng.random() < 0.6 else pose_open(rng, subject)
        path, orient, motion = traj_circle(rng, n)
    else:
        raise ValueError(label)
    local = local_landmarks(subject, [spec], np.zeros(n), rng)
    frames = project(local, view, path, orient, rng, subject.jitter * rng.uniform(0.7, 1.4))
    lid = LABEL_TO_ID[label]
    return {
        "frames": frames.astype(np.float32),
        "aspect": view.aspect,
        "label": lid,
        "frame_labels": np.full(n, lid, dtype=np.int8),
        "motion": motion,
        "kind": "single",
    }


TRANSITION_STATICS = tuple(STATIC_LABELS)


def generate_transition(a: str, b: str, rng: np.random.Generator, subject: Subject, n: int = CLIP_FRAMES) -> dict:
    """Static gesture A morphing into static gesture B, with per-frame labels (-1 = in between)."""
    view = sample_view(rng, b)
    spec_a = STATIC_POSES[a](rng, subject)
    spec_b = STATIC_POSES[b](rng, subject)
    start = int(rng.integers(4, n - 6))
    length = int(rng.integers(3, 8))
    t = np.arange(n)
    w = smoothstep((t - start) / length)
    local = local_landmarks(subject, [spec_a, spec_b], w, rng)
    path = traj_hover(rng, n, max_speed=3.0) if rng.random() < 0.6 else traj_drift(rng, n, rng.uniform(0.03, 0.3))
    orient = np.zeros((n, 3))
    orient[:, 1] = np.radians(rng.uniform(-15, 15)) * w
    frames = project(local, view, path, orient, rng, subject.jitter * rng.uniform(0.7, 1.4))
    frame_labels = np.where(t < start, LABEL_TO_ID[a], np.where(t < start + length, -1, LABEL_TO_ID[b])).astype(np.int8)
    return {
        "frames": frames.astype(np.float32),
        "aspect": view.aspect,
        "label": LABEL_TO_ID[b],
        "frame_labels": frame_labels,
        "motion": np.zeros(n, dtype=bool),
        "kind": "transition",
    }


def generate_dataset(
    n_per_class: int = 1000,
    n_transitions: int = 2500,
    n_subjects: int = 48,
    seed: int = 0,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    subjects = [sample_subject(rng) for _ in range(n_subjects)]
    clips: list[dict] = []
    for label in LABELS:
        for _ in range(n_per_class):
            sid = int(rng.integers(n_subjects))
            clip = generate_clip(label, rng, subjects[sid])
            clip.update(group=f"synth-{sid:03d}", source="synthetic")
            clips.append(clip)
    pairs = [(a, b) for a in TRANSITION_STATICS for b in TRANSITION_STATICS if a != b]
    weights = np.array([2.0 if "pinch_draw" in (a, b) else 1.0 for a, b in pairs])
    weights /= weights.sum()
    for _ in range(n_transitions):
        a, b = pairs[int(rng.choice(len(pairs), p=weights))]
        sid = int(rng.integers(n_subjects))
        clip = generate_transition(a, b, rng, subjects[sid])
        clip.update(group=f"synth-{sid:03d}", source="synthetic")
        clips.append(clip)
    return clips


def _heart(n: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    pts = np.stack([x, y], -1)
    return pts / 32.0  


def _star(points: int = 5) -> np.ndarray:
    ang = -np.pi / 2 + np.arange(2 * points + 1) * np.pi / points
    rad = np.where(np.arange(2 * points + 1) % 2 == 0, 1.0, 0.42)
    return np.stack([rad * np.cos(ang), rad * np.sin(ang)], -1)


def _resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    s = np.concatenate([[0], np.cumsum(seg)])
    target = np.linspace(0, s[-1], n)
    return np.stack([np.interp(target, s, pts[:, d]) for d in range(2)], -1)


def generate_demo(seed: int = 7, aspect: float = 16 / 9) -> dict:
    """Scripted ~24 s session. The *cursor* path is designed in screen space and
    the wrist is solved from it, so shapes come out exactly as drawn."""
    rng = np.random.default_rng(seed)
    subject = sample_subject(np.random.default_rng(3))
    subject = Subject(subject.mcp, subject.phalanx, subject.thumb_cmc, subject.thumb_len, 0.0, 0.006)

    def scr(x: float, y: float) -> np.ndarray:  
        return np.array([(1.0 - x) * aspect, y])

    def hold(p: np.ndarray, frames: int) -> np.ndarray:
        return np.repeat(p[None], frames, 0)

    def move(p0: np.ndarray, p1: np.ndarray, frames: int) -> np.ndarray:
        s = min_jerk(np.linspace(0, 1, frames))
        return p0[None] + s[:, None] * (p1 - p0)[None]

    def glide(p0: np.ndarray, p1: np.ndarray, peak: float = 2.8) -> np.ndarray:
        """Reposition at a human pace: peak speed ``peak`` palm lengths per second."""
        dist = np.linalg.norm(p1 - p0) / 0.17
        return move(p0, p1, max(12, int(np.ceil(FPS * 1.875 * dist / peak))))

    def shape(pts_screen: np.ndarray, frames: int) -> np.ndarray:
        pts = np.stack([scr(x, y) for x, y in pts_screen])
        return _resample_polyline(pts, frames)

    heart = _heart(200) * np.array([0.26 / aspect, 0.26]) + np.array([0.33, 0.43])
    star = _resample_polyline(_star(), 200) * np.array([0.14 / aspect, 0.14]) + np.array([0.67, 0.42])
    zz_x = np.linspace(0.42, 0.58, 9)
    zigzag = np.stack([zz_x, 0.78 + 0.035 * np.where(np.arange(9) % 2 == 0, -1, 1)], -1)
    wave_x = np.linspace(0.24, 0.78, 120)
    wave = np.stack([wave_x, 0.7 + 0.025 * np.sin((wave_x - 0.24) / 0.54 * 4 * np.pi)], -1)

    segs: list[tuple[str, np.ndarray, str]] = []
    cur = scr(0.62, 0.95)

    def add(kind: str, path: np.ndarray, anchor: str = "pinch") -> None:
        nonlocal cur
        segs.append((kind, path, anchor))
        cur = path[-1]

    add("open", glide(cur, scr(heart[0, 0], heart[0, 1])))
    add("open", hold(cur, 8))
    add("pinch", shape(heart, 96))
    add("open", glide(cur, scr(0.52, 0.32)))
    add("peace", hold(cur, 32))
    add("open", glide(cur, scr(star[0, 0], star[0, 1])))
    add("open", hold(cur, 6))
    add("pinch", shape(star, 96))
    add("open", glide(cur, scr(zigzag[0, 0], zigzag[0, 1])))
    add("open", hold(cur, 6))
    add("pinch", shape(zigzag, 36))
    add("open", glide(cur, scr(0.62, 0.55)))
    add("open", hold(cur, 10))
    add("swipe", move(cur, scr(0.3, 0.57), 11))
    add("open", hold(cur, 20))
    add("peace", hold(cur, 32))
    add("open", glide(cur, scr(wave[0, 0], wave[0, 1])))
    add("open", hold(cur, 6))
    add("pinch", shape(wave, 72))
    add("open", glide(cur, scr(0.66, 0.70)))
    fist_path = np.concatenate([move(scr(0.66, 0.70), scr(0.8, 0.70), 14), move(scr(0.8, 0.70), scr(0.66, 0.71), 14), move(scr(0.66, 0.71), scr(0.8, 0.69), 14)])
    add("fist", fist_path, anchor="palm")
    add("open", glide(cur, scr(0.5, 0.45)))
    add("open", hold(cur, 14))
    circle_t = np.linspace(0, 1, 22)
    loop_ang = np.pi / 2 + 2 * np.pi * smoothstep(circle_t) * 1.05
    circ = np.stack([scr(0.5 + 0.085 * np.cos(a) / aspect, 0.45 - 0.1 + 0.1 * np.sin(a)) for a in loop_ang])
    circ += cur - circ[0]
    add("point", circ)
    add("open", hold(cur, 40))

    poses = {"open": pose_open, "pinch": pose_pinch, "peace": pose_peace, "fist": pose_fist, "point": pose_point, "swipe": pose_open}
    seg_specs = []
    for kind, _, _ in segs:
        spec = poses[kind](rng, subject)
        if kind in ("open", "swipe"):
            spec.thumb["abd"] = np.radians(40)
        seg_specs.append(spec)

    total = sum(len(p) for _, p, _ in segs)
    cursor = np.concatenate([p for _, p, _ in segs])
    anchors = np.concatenate([[a] * len(p) for _, p, a in segs])
    local = np.zeros((total, 21, 3))
    blend = 5
    start = 0
    for i, (kind, path, _) in enumerate(segs):
        n = len(path)
        prev = seg_specs[i - 1] if i > 0 else seg_specs[i]
        w = smoothstep(np.arange(n) / blend) if i > 0 else np.ones(n)
        local[start : start + n] = local_landmarks(subject, [prev, seg_specs[i]], w, rng)
        start += n

    view = View(roll=np.radians(-8), yaw=np.radians(18), pitch=np.radians(-10), left=False, scale=0.17, aspect=aspect, center=np.zeros(2), zscale=0.9)
    orient = np.zeros((total, 3))
    rel = project(local, view, np.zeros((total, 2)), orient, rng, jitter=subject.jitter, orient_noise=np.radians(1.5))
    rel_iso = rel.copy()
    rel_iso[..., 0] *= aspect
    pinch_pt = 0.5 * (rel_iso[:, 4, :2] + rel_iso[:, 8, :2])
    palm_pt = rel_iso[:, [0, 5, 9, 13, 17], :2].mean(axis=1)
    anchor_off = np.where((anchors == "palm")[:, None], palm_pt, pinch_pt)
    kernel = np.ones(5) / 5
    anchor_off = np.stack([np.convolve(np.pad(anchor_off[:, d], 2, mode="edge"), kernel, mode="valid") for d in range(2)], -1)
    wrist = cursor - anchor_off
    frames = rel_iso.copy()
    frames[..., :2] += wrist[:, None, :]
    frames[..., 0] /= aspect
    return {
        "format": "air-canvas-replay",
        "version": 1,
        "fps": FPS,
        "aspect": aspect,
        "handedness": "Right",
        "synthetic": True,
        "frames": frames.reshape(total, 63).astype(np.float32),
        "script": [(kind, int(len(p))) for kind, p, _ in segs],
    }
