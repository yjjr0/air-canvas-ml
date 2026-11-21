"""The feature spec is the contract between training and the browser."""

import json

import numpy as np
import pytest

from dataset import crop_label, features_from_isotropic, load_recordings, make_windows, resample_time, splice, to_isotropic, window_features
from gestures import FEATURES, LABEL_TO_ID, WINDOW
from synth import generate_clip, generate_transition, sample_subject


@pytest.fixture(scope="module")
def clip():
    rng = np.random.default_rng(0)
    return generate_clip("swipe_undo", rng, sample_subject(rng))


def test_shape_and_dtype(clip):
    f = window_features(clip["frames"][:WINDOW], clip["aspect"])
    assert f.shape == (FEATURES, WINDOW)
    assert f.dtype == np.float32


def test_translation_and_scale_invariance(clip):
    w = clip["frames"][:WINDOW].astype(np.float64)
    a = window_features(w, clip["aspect"])
    moved = w.copy()
    moved[..., 0] += 0.1
    moved[..., 1] -= 0.07
    assert np.allclose(window_features(moved, clip["aspect"]), a, atol=1e-5)
    centre = w[0, 0].copy()
    zoomed = centre + (w - centre) * 1.7
    zoomed[..., 2] = w[..., 2] * 1.7
    assert np.allclose(window_features(zoomed, clip["aspect"]), a, atol=1e-4)


def test_wrist_position_is_zero_and_velocity_starts_at_zero(clip):
    f = window_features(clip["frames"][:WINDOW], clip["aspect"])
    assert np.all(f[0:3] == 0)
    assert np.all(f[63:, 0] == 0)


def test_velocity_keeps_whole_hand_motion(clip):
    """Wrist-centring alone would erase a swipe; velocities must carry it."""
    feats, _ = make_windows([clip], None, augment=False)
    vel_x = feats[0, 63::3]
    assert np.abs(vel_x).max() > 0.2


def test_aspect_correction_makes_coordinates_isotropic(clip):
    w = clip["frames"][:WINDOW].astype(np.float64)
    p = to_isotropic(w, 16 / 9)
    assert np.allclose(p[..., 0], w[..., 0] * 16 / 9)
    assert np.allclose(p[..., 1], w[..., 1])


def test_batch_and_single_agree(clip):
    w = clip["frames"][:WINDOW]
    single = window_features(w, clip["aspect"])
    batch = features_from_isotropic(to_isotropic(np.stack([w, w]), clip["aspect"]))
    assert np.array_equal(batch[0], single) and np.array_equal(batch[1], single)


def test_resample_time_endpoints():
    x = np.arange(24, dtype=np.float64)[:, None, None] * np.ones((1, 21, 3))
    y = resample_time(x, WINDOW)
    assert y.shape == (WINDOW, 21, 3)
    assert y[0, 0, 0] == 0 and y[-1, 0, 0] == 23


def test_transition_labels_follow_the_end_of_the_window():
    rng = np.random.default_rng(3)
    clip = generate_transition("open_palm_hover", "pinch_draw", rng, sample_subject(rng))
    fl = clip["frame_labels"]
    start_b = int(np.argmax(fl == LABEL_TO_ID["pinch_draw"]))
    first_mixed = int(np.argmax(fl == -1))
    assert crop_label(clip, 0, first_mixed) == LABEL_TO_ID["open_palm_hover"]
    assert crop_label(clip, 0, start_b + 1) == LABEL_TO_ID["idle"]
    assert crop_label(clip, len(fl) - WINDOW, WINDOW) == LABEL_TO_ID["pinch_draw"]


def test_recordings_roundtrip(tmp_path, clip):
    doc = {
        "format": "air-canvas-clips",
        "version": 1,
        "session": "s-test",
        "clips": [
            {"label": "fist_erase", "aspect": 1.7778, "session": "s-test", "frames": clip["frames"].reshape(len(clip["frames"]), 63).tolist()},
            {"label": "my_custom", "aspect": 1.7778, "frames": clip["frames"].reshape(len(clip["frames"]), 63).tolist()},
        ],
    }
    (tmp_path / "rec.json").write_text(json.dumps(doc))
    clips = load_recordings(tmp_path)
    assert len(clips) == 1 and clips[0]["label"] == LABEL_TO_ID["fist_erase"] and clips[0]["group"] == "s-test"


def test_splice_has_no_teleport():
    rng = np.random.default_rng(4)
    subj = sample_subject(rng)
    a = generate_clip("open_palm_hover", rng, subj)
    b = generate_clip("pinch_draw", rng, subj)
    for c in (a, b):
        c["group"] = "g"
    s = splice(a, b, rng)
    p = to_isotropic(s["frames"], s["aspect"])
    palm = np.median(np.linalg.norm(p[:, 9] - p[:, 0], axis=-1))
    wrist_step = np.linalg.norm(np.diff(p[:, 0, :2], axis=0), axis=-1) / palm
    assert wrist_step.max() < 0.6  
