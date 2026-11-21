"""Sanity checks that the synthetic generator produces what each label claims."""

import numpy as np
import pytest

from dataset import motion_energy
from gestures import CLIP_FRAMES, LABELS
from synth import generate_clip, generate_demo, generate_transition, sample_subject

THUMB_TIP, INDEX_TIP = 4, 8


def tip_gap(clip):
    p = clip["frames"].astype(np.float64).copy()
    p[..., 0] *= clip["aspect"]
    p[..., 2] *= clip["aspect"]
    palm = np.linalg.norm(p[:, 9] - p[:, 0], axis=-1)
    return np.median(np.linalg.norm(p[:, THUMB_TIP] - p[:, INDEX_TIP], axis=-1) / palm)


@pytest.mark.parametrize("label", LABELS)
def test_clip_shapes(label):
    rng = np.random.default_rng(1)
    clip = generate_clip(label, rng, sample_subject(rng))
    assert clip["frames"].shape == (CLIP_FRAMES, 21, 3)
    assert np.isfinite(clip["frames"]).all()
    assert clip["frame_labels"].shape == (CLIP_FRAMES,)


def test_pinch_closes_thumb_and_index_but_idle_does_not():
    rng = np.random.default_rng(2)
    pinch = [tip_gap(generate_clip("pinch_draw", rng, sample_subject(rng))) for _ in range(40)]
    idle = [tip_gap(generate_clip("idle", rng, sample_subject(rng))) for _ in range(40)]
    assert np.median(pinch) < 0.2
    assert np.percentile(idle, 10) > np.percentile(pinch, 90)


def test_swipes_move_much_faster_than_hovering():
    rng = np.random.default_rng(3)
    peak = lambda label: np.median([motion_energy(generate_clip(label, rng, sample_subject(rng))).max() for _ in range(40)])
    assert peak("swipe_undo") > 2 * peak("open_palm_hover")


def test_transition_has_both_labels():
    rng = np.random.default_rng(4)
    clip = generate_transition("open_palm_hover", "pinch_draw", rng, sample_subject(rng))
    assert {-1, 1, 2} <= set(clip["frame_labels"].tolist())


def test_demo_is_deterministic():
    a, b = generate_demo(), generate_demo()
    assert np.array_equal(a["frames"], b["frames"])
    assert a["frames"].shape[1] == 63
