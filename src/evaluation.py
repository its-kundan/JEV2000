import argparse
import platform
import time

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch

from src.common import LABELS, OUT, SEED, device_run, fingerprint, read, write_json
from src.inference import FILES, NAMES, InferenceModel, available


def evaluate(mode, test, device, repeats):
    started = time.perf_counter()
    model = InferenceModel(mode, device)
    load_seconds = time.perf_counter() - started
    if model.run["data"] != fingerprint("train", "validation"):
        raise ValueError(f"{mode} was trained on different data. Retrain before evaluating.")
    texts = test.text.tolist()
    batches = [model.probabilities(texts[start:start + 32]) for start in range(0, len(texts), 32)]
    scores = np.concatenate(batches)
    predictions = np.asarray(LABELS)[scores.argmax(axis=1)]
    report = classification_report(test.label, predictions, labels=LABELS, output_dict=True, zero_division=0)
    samples = [texts[i] for i in np.linspace(0, len(texts) - 1, 24, dtype=int)]
    for text in samples[:5]:
        model.probabilities([text])
    times = []
    for _ in range(repeats):
        for text in samples:
            start = time.perf_counter()
            model.probabilities([text])  # CPU materialization synchronizes accelerator outputs
            times.append((time.perf_counter() - start) * 1000)
    folder = OUT / mode
    bundle = set(FILES[mode])
    encoder = folder / "encoder" if mode == "jepa" else folder
    bundle.update(str(path.relative_to(folder)) for path in encoder.iterdir()
                  if path.name in {"tokenizer_config.json", "special_tokens_map.json", "vocab.txt"})
    bundle.add("run.json")
    training = model.run
    result = dict(model=NAMES[mode], accuracy=accuracy_score(test.label, predictions),
                  macro_f1=report["macro avg"]["f1-score"], macro_precision=report["macro avg"]["precision"],
                  macro_recall=report["macro avg"]["recall"], per_class={k: report[k] for k in LABELS},
                  confusion=confusion_matrix(test.label, predictions, labels=LABELS).tolist(),
                  latency_median_ms=float(np.median(times)), latency_p95_ms=float(np.percentile(times, 95)),
                  latency_measurements=len(times), load_seconds=load_seconds,
                  parameters=training["trainable_parameters"], training_seconds=training["seconds"],
                  training_device=training["device"], inference_device=str(model.device),
                  inference_threads=torch.get_num_threads(),
                  bytes=sum((folder / path).stat().st_size for path in bundle),
                  smoke=training.get("smoke", False), training_run=training)
    if mode == "jepa":
        result.update(pretraining_parameters=training["pretrain"]["trainable_parameters"],
                      pretraining_seconds=training["pretrain"]["seconds"],
                      pretraining_device=training["pretrain"]["device"])
    errors = test.loc[test.label != predictions, ["ticket_id", "text", "label"]].assign(
        predicted=predictions[test.label != predictions],
        confidence=scores.max(axis=1)[test.label != predictions])
    return result, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--models", nargs="+", choices=NAMES, default=list(NAMES))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("Repeats must be positive")
    missing = [NAMES[mode] for mode in args.models if not available(mode)]
    if missing:
        parser.error("Missing completed checkpoints for " + ", ".join(missing))
    test = read("test")
    folder = OUT / "metrics"
    folder.mkdir(parents=True, exist_ok=True)
    result = {}
    for mode in args.models:
        result[mode], errors = device_run(lambda device: evaluate(mode, test, device, args.repeats),
                                           args.device, SEED)
        errors.to_csv(folder / f"{mode}_errors.csv", index=False)
        print(f"{mode}: accuracy={result[mode]['accuracy']:.3f} macro F1={result[mode]['macro_f1']:.3f}", flush=True)
    write_json(folder / "benchmark.json", dict(data=fingerprint("train", "validation", "test"),
               hardware=dict(system=platform.platform(), machine=platform.machine()), test_size=len(test),
               labels=LABELS, results=result))


if __name__ == "__main__":
    main()
