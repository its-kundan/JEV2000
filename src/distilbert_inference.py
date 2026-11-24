import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common import load_json
from src.confidence_calibration import probabilities
from src.router_config import DISTILBERT


class DistilBERT:
    def __init__(self, path=DISTILBERT, device="cpu"):
        if device == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        self.run = load_json(path / "run.json")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).to(device).eval()
        self.device = device
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.name = ("Pretrained DistilBERT + random head" if path.name.startswith("distilbert_initial")
                     else "Fine-tuned DistilBERT")
        self.temperature = self.run["temperature"]
        self.seconds = self.run["seconds"]
        self.size = sum(file.stat().st_size for file in path.iterdir() if file.is_file())

    @torch.inference_mode()
    def predict(self, texts, calibrated=True):
        outputs = []
        texts = list(texts)
        for start in range(0, len(texts), 32):
            batch = self.tokenizer(texts[start:start + 32], padding=True, truncation=True,
                                   max_length=self.run["config"]["max_length"], return_tensors="pt")
            outputs.append(self.model(**batch.to(self.device)).logits.cpu().numpy())
        return probabilities(np.concatenate(outputs), self.temperature if calibrated else 1)
