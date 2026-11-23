import numpy as np
import pytest
import torch

from src import baseline, inference
from src.common import LABELS


def test_pipeline_reload_and_probability_alignment(tmp_path, monkeypatch):
    from data.generate_dataset import build

    data = build()
    monkeypatch.setattr(baseline, "OUT", tmp_path)
    monkeypatch.setattr(inference, "OUT", tmp_path)
    monkeypatch.setattr(baseline, "read", lambda split: data[data.split.eq(split)])
    monkeypatch.setattr(baseline, "fingerprint", lambda *splits: dict.fromkeys(splits, "test"))
    assert not inference.available("baseline")
    with pytest.raises(FileNotFoundError):
        inference.InferenceModel("baseline", torch.device("cpu"))
    baseline.main()
    model = inference.InferenceModel("baseline", torch.device("cpu"))
    text = "I was charged twice and want my money back."
    scores = model.probabilities([text])[0]
    assert scores.shape == (len(LABELS),)
    assert scores.sum() == pytest.approx(1.)
    result = model.ticket(text, 0)
    assert result["category"] == LABELS[np.argmax(scores)]
    assert result["category"] == "Refund"
    assert result["route"] == "Refund Operations"
    assert model.ticket(text, 1)["route"] == "Human Review Queue"
    assert len(result["top"]) == 3 and result["latency_ms"] >= 0
    with pytest.raises(ValueError):
        model.ticket("  ", .5)
