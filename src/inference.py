"""Load completed, local checkpoints and return probabilities in LABELS order."""
import time

import joblib
import numpy as np
from scipy.special import softmax
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common import LABELS, OUT, load_json
from src.linear_probe import FrozenEncoder
from src.router import destination

NAMES = {"baseline": "TF-IDF + Logistic Regression", "distilbert": "Fine-tuned DistilBERT",
         "jepa": "Text JEPA + Linear Probe", "reference": "Pretrained Encoder + Linear Probe"}
FILES = {"baseline": ("pipeline.joblib",), "distilbert": ("model.safetensors", "config.json", "tokenizer.json"),
         "jepa": ("encoder/model.safetensors", "encoder/config.json", "encoder/tokenizer.json", "probe.joblib"),
         "reference": ("model.safetensors", "config.json", "tokenizer.json", "probe.joblib")}


def available(mode):
    folder = OUT / mode
    if not all((folder / name).is_file() for name in (*FILES[mode], "run.json")):
        return False
    if mode in {"jepa", "reference"}:
        path = OUT / "jepa/pretrain.json"
        return path.is_file() and load_json(path) == load_json(folder / "run.json").get("pretrain")
    return True


class InferenceModel:
    def __init__(self, mode, device):
        if not available(mode):
            raise FileNotFoundError(f"Run the training CLI for {NAMES[mode]}")
        self.mode = mode
        self.device = torch.device("cpu") if mode == "baseline" else device
        folder = OUT / mode
        self.run = load_json(folder / "run.json")
        if mode == "baseline":
            self.model = joblib.load(folder / "pipeline.joblib")
        elif mode == "distilbert":
            self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                folder, local_files_only=True, attn_implementation="eager").to(device).eval()
        else:
            self.encoder = FrozenEncoder(folder / "encoder" if mode == "jepa" else folder,
                                         self.run["max_length"], device)
            self.model = joblib.load(folder / "probe.joblib")

    @torch.inference_mode()
    def probabilities(self, texts):
        if self.mode == "baseline":
            order = [list(self.model.classes_).index(label) for label in LABELS]
            return self.model.predict_proba(texts)[:, order]
        if self.mode == "distilbert":
            batch = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt",
                                   max_length=self.run["config"]["max_length"]).to(self.device)
            return self.model(**batch).logits.softmax(-1).cpu().numpy()
        logits = self.model.decision_function(self.encoder.embed(texts))
        return softmax(logits / self.run["temperature"], axis=1)

    def ticket(self, text, threshold, departments=None):
        if not text.strip():
            raise ValueError("Enter a customer-support ticket")
        started = time.perf_counter()
        scores = self.probabilities([text.strip()])[0]
        latency = (time.perf_counter() - started) * 1000
        top = [(LABELS[i], float(scores[i])) for i in np.argsort(-scores)[:3]]
        label, confidence = top[0]
        return dict(model=NAMES[self.mode], category=label, confidence=confidence, top=top,
                    latency_ms=latency, route=destination(label, confidence, threshold, departments),
                    device=str(self.device))
