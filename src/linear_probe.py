import argparse
import time

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoTokenizer, DistilBertModel

from src.common import LABELS, OUT, SEED, device_run, fingerprint, load_json, read, write_json
from src.jepa_model import mean_pool


class FrozenEncoder:
    def __init__(self, path, max_length, device):
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = DistilBertModel.from_pretrained(
            path, local_files_only=True, attn_implementation="eager").to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.max_length = max_length

    @torch.inference_mode()
    def embed(self, texts, batch_size=32):
        vectors = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt",
                                    return_special_tokens_mask=True).to(self.device)
            special = tokens.pop("special_tokens_mask")
            visible = tokens.attention_mask.bool()
            ordinary = visible & ~special.bool()
            mask = torch.where(ordinary.any(dim=1, keepdim=True), ordinary, visible)
            vectors.append(mean_pool(self.model(**tokens).last_hidden_state, mask).cpu().numpy())
        return np.concatenate(vectors)


def calibrate_temperature(logits, gold):
    indices = np.arange(len(gold))

    def nll(log_temperature):
        return -log_softmax(logits / np.exp(log_temperature), axis=1)[indices, gold].mean()

    result = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    if not result.success:
        raise RuntimeError("Validation temperature fitting failed")
    return float(np.exp(result.x))


def fit_probe(mode, device):
    started = time.perf_counter()
    pretrain = load_json(OUT / "jepa/pretrain.json")
    if pretrain["data"] != fingerprint("train"):
        raise ValueError("Retrain JEPA for the current training split")
    max_length = pretrain["config"]["max_length"]
    folder = OUT / mode
    encoder = FrozenEncoder(folder / "encoder" if mode == "jepa" else folder, max_length, device)
    train, val = read("train"), read("validation")
    x_train, x_val = encoder.embed(train.text.tolist()), encoder.embed(val.text.tolist())
    y_train, y_val = train.label.map(LABELS.index), val.label.map(LABELS.index)
    trials, best = [], None
    for c in (.01, .1, 1.):
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=SEED))
        probe.fit(x_train, y_train)
        f1 = f1_score(y_val, probe.predict(x_val), average="macro")
        trials.append(dict(C=c, validation_macro_f1=f1))
        if best is None or f1 > best[0]:
            best = f1, probe, c
    _, probe, c = best
    temperature = calibrate_temperature(probe.decision_function(x_val), y_val.to_numpy())
    joblib.dump(probe, folder / "probe.joblib", compress=3)
    if mode == "jepa":
        coords = PCA(n_components=2, random_state=SEED).fit(x_train).transform(x_val)
        val.assign(pc1=coords[:, 0], pc2=coords[:, 1]).to_csv(folder / "pca_validation.csv", index=False)
    head = probe[-1]
    write_json(folder / "run.json", dict(mode=mode, C=c, temperature=temperature, max_length=max_length,
               trainable_parameters=head.coef_.size + head.intercept_.size, trials=trials,
               seconds=time.perf_counter() - started, device=str(device), pretrain=pretrain,
               smoke=pretrain["smoke"], data=fingerprint("train", "validation")))
    print(f"{mode}: validation macro F1={best[0]:.4f}; temperature={temperature:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    for mode in ("reference", "jepa"):
        device_run(lambda device: fit_probe(mode, device), args.device, SEED)


if __name__ == "__main__":
    main()
