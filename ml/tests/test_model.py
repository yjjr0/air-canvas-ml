"""Model contract: shapes, normalised embedding, ONNX export parity."""

import numpy as np
import pytest
import torch

from gestures import FEATURES, LABELS, WINDOW
from model import GestureNet1D, count_parameters


def test_outputs():
    model = GestureNet1D().eval()
    logits, emb = model(torch.randn(3, FEATURES, WINDOW))
    assert logits.shape == (3, len(LABELS))
    assert emb.shape == (3, 32)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_stays_small():
    assert count_parameters(GestureNet1D()) < 80_000


def test_recent_frames_matter():
    """The 'recent' pooling branch must make the output depend on frame order."""
    torch.manual_seed(0)
    model = GestureNet1D().eval()
    x = torch.randn(1, FEATURES, WINDOW)
    flipped = torch.flip(x, dims=[-1])
    assert not torch.allclose(model(x)[0], model(flipped)[0])


@pytest.mark.slow
def test_onnx_parity(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from train_export import export_onnx

    model = GestureNet1D().eval()
    sample = np.random.default_rng(0).normal(size=(8, FEATURES, WINDOW)).astype(np.float32)
    report = export_onnx(model, tmp_path / "m.onnx", sample)  
    assert report["random"]["max_abs_logit_diff"] < 1e-4
