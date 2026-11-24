import joblib
import numpy as np
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast

from data.generate_destinations import TAXONOMY
from data.generate_tickets import build_tickets
from src.baseline_model import Baseline, fit_baseline
from src.distilbert_inference import DistilBERT


def test_baseline_fit_and_reload(tmp_path):
    tickets = build_tickets(480)
    train, val = (tickets[tickets.split == split] for split in ("train", "validation"))
    path = tmp_path / "baseline.joblib"
    joblib.dump(fit_baseline(train, val), path)
    model = Baseline(path)
    scores = model.predict(["Refund my duplicate charge", "Help, login fails"])
    assert set(model.labels) == set(TAXONOMY)
    assert scores.shape == (2, 8)
    np.testing.assert_allclose(scores.sum(axis=1), 1)


def test_offline_distilbert_checkpoint_training_and_loading(tmp_path, monkeypatch):
    from src import train_distilbert

    base = tmp_path / "pretrained"
    base.mkdir()
    vocab = base / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nrefund\nplease\nmy\ncard\n")
    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(base)
    config = DistilBertConfig(vocab_size=9, n_layers=1, dim=24, hidden_dim=32, n_heads=2,
                              max_position_embeddings=32)
    DistilBertForSequenceClassification(config).distilbert.save_pretrained(base)
    destination = tmp_path / "distilbert"
    monkeypatch.setattr(train_distilbert, "DISTILBERT", destination)
    monkeypatch.setattr(train_distilbert, "INITIAL", tmp_path / "distilbert_initial")
    monkeypatch.setattr(train_distilbert, "LOGS", tmp_path)
    tickets = build_tickets(240)
    monkeypatch.setattr(train_distilbert, "read", lambda split: tickets[tickets.split == split]
                        .rename(columns={"ticket_text": "text", "department": "label"}))
    from argparse import Namespace

    torch.set_num_threads(2)
    train_distilbert.fit(Namespace(model=str(base), freeze_layers=0, batch_size=16, max_length=24,
                                  seed=42, lr=2e-5, weight_decay=.01, max_steps=1, epochs=1, patience=2),
                         torch.device("cpu"))
    model = DistilBERT(destination.with_name("distilbert_smoke"))
    assert model.run["smoke"]
    assert model.run["history"][0]["optimizer_steps"] == 1
    assert model.labels == list(TAXONOMY)
    scores = model.predict(["please refund my card"])
    assert scores.shape == (1, 8)
    np.testing.assert_allclose(scores.sum(), 1, rtol=1e-6)
