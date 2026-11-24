"""Evaluate local models on common held-out tickets and compare routing policies."""
import argparse
import hashlib
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from data.generate_destinations import CATALOG
from data.prepare_dataset import DATA, SPLITS, read_tickets
from src.baseline_model import Baseline
from src.candidate_retriever import Retriever
from src.common import write_json
from src.confidence_calibration import calibration_metrics, probabilities
from src.jev_simulator import Simulator
from src.router_config import BASELINE, CONFIG, DISTILBERT, INITIAL, METRICS
from src.routing_engine import route


def classifier_paths(include_initial=False):
    paths = {"Baseline": BASELINE, "DistilBERT": DISTILBERT}
    if include_initial:
        paths["Pretrained + random head"] = INITIAL
    return paths


def model_available(path):
    return path.is_file() if path == BASELINE else (
        (path / "run.json").is_file() and (path / "model.safetensors").is_file()
    )


def load_classifier(path):
    if path == BASELINE:
        return Baseline()
    from src.distilbert_inference import DistilBERT

    return DistilBERT(path)


def department_metrics(model, frame):
    labels = model.labels
    truth = frame.department.map(labels.index).to_numpy()
    original = model.predict(frame.ticket_text, calibrated=False)
    # log(p) differs from logits by a per-example constant, which softmax cancels.
    calibrated = probabilities(np.log(np.maximum(original, np.finfo(original.dtype).tiny)), model.temperature)
    predicted = calibrated.argmax(axis=1)
    sample = frame.ticket_text.iloc[:24]
    model.predict([sample.iloc[0]])
    timings = []
    for ticket in sample:
        start = perf_counter()
        model.predict([ticket])
        timings.append((perf_counter() - start) * 1000)
    report = classification_report(truth, predicted, labels=list(range(len(labels))),
                                   target_names=labels, output_dict=True, zero_division=0)
    return dict(
        report=report, labels=labels,
        confusion=confusion_matrix(truth, predicted, labels=range(len(labels))).tolist(),
        calibration={name: calibration_metrics(probs, truth) for name, probs in
                     (("Before scaling", original), ("After scaling", calibrated))},
        latency_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings, 95)),
        errors=[dict(ticket_id=row.ticket_id, ticket_text=row.ticket_text,
                     actual=model.labels[a], predicted=model.labels[b])
                for row, a, b in zip(frame.itertuples(), truth, predicted) if a != b][:25],
    )


def evaluate_routes(model, frame, retriever, backend, k, threshold, predictions=None):
    counts = dict(retrieved=0, proposed_correct=0, auto_correct=0, auto_count=0,
                  schema_failures=0, review_labels=0, review_label_caught=0)
    timings, decision_timings = [], []
    examples = list(frame.itertuples())
    for index, row in enumerate(examples):
        result = route(row.ticket_text, model, retriever, backend, k, threshold, row.region,
                       prediction=None if predictions is None else predictions[index])
        auto = bool(result.destination)
        counts["auto_count"] += auto
        counts["auto_correct"] += result.destination == row.destination_id
        counts["review_labels"] += row.requires_human_review
        counts["review_label_caught"] += row.requires_human_review and not auto
        counts["schema_failures"] += result.schema_failure
        timings.append(result.latency_ms)
        if result.state:
            counts["retrieved"] += row.destination_id in {c.destination_id for c in result.state.candidates}
        if result.decision:
            decision_timings.append(result.decision.latency_ms)
            counts["proposed_correct"] += result.decision.destination_id == row.destination_id
    n = len(examples)
    auto_count = counts["auto_count"]
    return dict(
        tickets=n, k=k, threshold=threshold, backend=backend.name,
        recall_at_k=counts["retrieved"] / n, proposed_accuracy=counts["proposed_correct"] / n,
        decision_accuracy_when_retrieved=(counts["proposed_correct"] / counts["retrieved"]
                                          if counts["retrieved"] else None),
        end_to_end_accuracy=counts["auto_correct"] / n, auto_coverage=auto_count / n,
        auto_accuracy=counts["auto_correct"] / auto_count if auto_count else None,
        human_review=1 - auto_count / n, misrouting_rate=(auto_count - counts["auto_correct"]) / n,
        review_label_recall=(counts["review_label_caught"] / counts["review_labels"]
                             if counts["review_labels"] else None),
        latency_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings, 95)),
        latency_scope="Retrieval + decision" if predictions is not None else "End to end",
        backend_latency_ms=float(np.median(decision_timings)) if decision_timings else None,
        schema_failures=counts["schema_failures"],
    )


def sample_tickets(frame, size):
    return frame.sample(n=min(size, len(frame)), random_state=CONFIG["seed"])


def data_signature():
    paths = [DATA / f"{split}.csv" for split in SPLITS] + [CATALOG]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Cap each split to this many seeded samples.")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    partitions = {name: read_tickets(name) for name in ("validation", "test")}
    if args.limit:
        partitions = {name: sample_tickets(frame, args.limit) for name, frame in partitions.items()}
    retriever = Retriever()
    metrics = dict(synthetic=True, dataset=data_signature(), models={}, routing=[], latency=[])
    backend = Simulator()
    for name, path in classifier_paths(include_initial=True).items():
        if (args.baseline_only and name != "Baseline") or not model_available(path):
            continue
        model = load_classifier(path)
        models = metrics["models"]
        models[name] = dict(name=model.name, size_bytes=model.size, training_seconds=model.seconds,
                            temperature=model.temperature, splits={
                                split: department_metrics(model, frame) for split, frame in partitions.items()
                            })
        if name == "Pretrained + random head":
            continue
        test = partitions["test"]
        predictions = model.predict(test.ticket_text)
        for k in [5, 10, 15, 25, 50]:
            result = evaluate_routes(model, test, retriever, backend, k, CONFIG["confidence_threshold"], predictions)
            metrics["routing"].append(dict(model=name, **result))
        warm = test.iloc[0]
        route(warm.ticket_text, model, retriever, backend, CONFIG["top_k"], region=warm.region)
        result = evaluate_routes(model, sample_tickets(test, 24), retriever, backend,
                                 CONFIG["top_k"], CONFIG["confidence_threshold"])
        metrics["latency"].append(dict(model=name, **result))
        print(f"Evaluated {name} on {len(test):,} test tickets.", flush=True)
    output = METRICS / "benchmark.json"
    write_json(output, metrics)
    print(f"Saved measured synthetic-data results to {output}")


if __name__ == "__main__":
    main()
