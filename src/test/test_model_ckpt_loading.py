from pathlib import Path

import pytest
import torch

from network.lstm_frame_srv import LSTMFrameSRV
import make_appleseed_lstm_predictors as target


REAL_CKPT = Path(r"C:\linux_project\LENS\runs\librispeech_LSTM_WORD_baseline_s000\ckpt\last.pt")


@pytest.mark.integration
@pytest.mark.skipif(not REAL_CKPT.exists(), reason="Real checkpoint not found")
def test_real_ckpt_can_be_loaded(monkeypatch):
    monkeypatch.setattr(target, "DEVICE", "cpu")
    monkeypatch.setattr(target, "CKPT_PATH", REAL_CKPT)

    model = target.load_model()

    assert isinstance(model, LSTMFrameSRV)
    assert model.training is False


@pytest.mark.integration
@pytest.mark.skipif(not REAL_CKPT.exists(), reason="Real checkpoint not found")
def test_real_ckpt_can_run_forward(monkeypatch):
    monkeypatch.setattr(target, "DEVICE", "cpu")
    monkeypatch.setattr(target, "CKPT_PATH", REAL_CKPT)

    model = target.load_model()

    x = torch.randn(1, 20, target.IN_DIM)
    x_lens = torch.tensor([20], dtype=torch.long)

    with torch.inference_mode():
        _, out, out_lens = model.forward_with_hidden(x, x_lens)

    assert out.ndim == 3
    assert out.shape[0] == 1
    assert out.shape[1] == 20
    assert out_lens.tolist() == [20]