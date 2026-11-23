import json

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_model
from transformers import BertTokenizerFast, DistilBertConfig, DistilBertModel

from src import linear_probe, train_jepa
from src.common import LABELS
from src.jepa_model import TextJEPA, masked_batch, objective


def config():
    return DistilBertConfig(vocab_size=32, max_position_embeddings=32, n_layers=1,
                           dim=16, hidden_dim=32, n_heads=2, dropout=0., attention_dropout=0.,
                           attn_implementation="eager")


def batch():
    ids = torch.tensor([[2, 5, 6, 7, 8, 3], [2, 9, 10, 11, 3, 0]])
    attention = ids.ne(0)
    special = ids.le(4)
    return ids, attention, special


def mask(ids, attention, special):
    return masked_batch(ids, attention, special, 4, torch.Generator().manual_seed(9), .5)


def test_masking_blocks_hidden_content():
    ids, attention, special = batch()
    context, positions, active = mask(ids, attention, special)
    changed = ids.clone()
    rows, columns = active.nonzero(as_tuple=True)
    selected = positions[rows, columns]
    changed[rows, selected] = 20
    remasked, again, same_active = mask(changed, attention, special)
    torch.testing.assert_close(context, remasked)
    torch.testing.assert_close(again, positions)
    torch.testing.assert_close(same_active, active)
    assert torch.all(context[rows, selected] == 4)
    visible = torch.ones_like(attention)
    visible[rows, selected] = False
    torch.testing.assert_close(context[visible], ids[visible])
    assert all(torch.diff(row[valid]).eq(1).all() for row, valid in zip(positions, active))


def test_teacher_gradients_ema_and_latent_updates():
    torch.manual_seed(7)
    model = TextJEPA(DistilBertModel(config()), 16).train()
    ids, attention, special = batch()
    context, positions, active = mask(ids, attention, special)
    output, target, pooled = model(context, ids, attention, positions)
    assert output.shape == target.shape == (*positions.shape, 16)
    assert pooled.shape == (2, 16)
    assert not target.requires_grad and not model.target.training
    loss, latent, spread = objective(output, target, active, pooled)
    assert torch.isfinite(torch.stack([loss, latent, spread])).all()
    student = list(model.context.parameters())
    teacher = list(model.target.parameters())
    student_before = [p.detach().clone() for p in student]
    teacher_before = [p.detach().clone() for p in teacher]
    head_before = model.predictor[-1].weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    loss.backward()
    assert all(p.grad is None for p in teacher)
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(student_before, student))
    assert not torch.equal(head_before, model.predictor[-1].weight)
    for old, new in zip(teacher_before, teacher):
        torch.testing.assert_close(old, new)
    model.update_target(.9)
    for old, new, learned in zip(teacher_before, teacher, student):
        torch.testing.assert_close(new, old.lerp(learned.detach(), .1))
    assert any(not torch.equal(old, new) for old, new in zip(teacher_before, teacher))


def test_jepa_cli_smoke_and_checkpoint(tmp_path, monkeypatch):
    model_path = tmp_path / "tiny"
    DistilBertModel(config()).save_pretrained(model_path)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "some", "test", "words"]))
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(model_path)
    def read_train(split):
        assert split == "train"
        return pd.DataFrame({"text": ["some test words", "test words some"]})

    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(train_jepa, "OUT", artifacts)
    monkeypatch.setattr(train_jepa, "read", read_train)
    monkeypatch.setattr(train_jepa, "fingerprint", lambda *splits: dict.fromkeys(splits, "smoke"))
    monkeypatch.setattr("sys.argv", ["train_jepa", "--model", str(model_path), "--layers", "1",
                                    "--epochs", "1", "--max-steps", "1", "--batch-size", "2",
                                    "--max-length", "16", "--device", "cpu"])
    train_jepa.main()
    data = json.loads((artifacts / "jepa/pretrain.json").read_text())
    assert data["smoke"] and data["data"] == {"train": "smoke"}
    checkpoint = artifacts / "jepa"
    restored = TextJEPA(DistilBertModel(config()), 16)
    load_model(restored, checkpoint / "research/jepa.safetensors")
    encoder = DistilBertModel.from_pretrained(checkpoint / "encoder")
    for full, saved in zip(restored.context.parameters(), encoder.parameters()):
        torch.testing.assert_close(full, saved)
    assert all(not p.requires_grad for p in restored.target.parameters())


def test_probe_fits_only_training_labels(tmp_path, monkeypatch):
    folder = tmp_path / "jepa"
    folder.mkdir()
    (folder / "pretrain.json").write_text(json.dumps({
        "config": {"max_length": 16}, "data": {"train": "ok"}, "smoke": True}))
    calls = []
    fitted = []
    real_fit = linear_probe.LogisticRegression.fit

    def checked_fit(self, x, y):
        assert len(x) == len(y) == 16
        fitted.append(y.tolist())
        return real_fit(self, x, y)

    class DummyEncoder:
        def __init__(self, *args):
            pass

        def embed(self, texts):
            return np.eye(8)[[int(text[-1]) for text in texts]]

    def read(split):
        calls.append(split)
        assert split in ("train", "validation")
        size = 2 if split == "train" else 1
        indices = list(range(len(LABELS))) * size
        return pd.DataFrame({"label": [LABELS[i] for i in indices],
                             "text": [f"{split}{i}" for i in indices]})

    monkeypatch.setattr(linear_probe, "OUT", tmp_path)
    monkeypatch.setattr(linear_probe, "fingerprint", lambda *splits: dict.fromkeys(splits, "ok"))
    monkeypatch.setattr(linear_probe, "read", read)
    monkeypatch.setattr(linear_probe, "FrozenEncoder", DummyEncoder)
    monkeypatch.setattr(linear_probe.LogisticRegression, "fit", checked_fit)
    linear_probe.fit_probe("jepa", torch.device("cpu"))
    assert calls == ["train", "validation"]
    assert fitted == [list(range(8)) * 2] * 3
    assert (folder / "probe.joblib").exists()
