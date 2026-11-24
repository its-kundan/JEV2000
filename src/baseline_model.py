import argparse
from time import perf_counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from data.prepare_dataset import read_tickets
from src.confidence_calibration import fit_temperature, probabilities
from src.router_config import BASELINE, CONFIG


def fit_baseline(train, validation):
    start = perf_counter()
    pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                             LogisticRegression(max_iter=400, random_state=CONFIG["seed"]))
    pipeline.fit(train.ticket_text, train.department)
    labels = list(pipeline.classes_)
    temperature = fit_temperature(pipeline.decision_function(validation.ticket_text),
                                  validation.department.map(labels.index).to_numpy())
    return dict(pipeline=pipeline, labels=labels, temperature=temperature, seconds=perf_counter() - start)


class Baseline:
    name = "TF-IDF + Logistic Regression"

    def __init__(self, path=BASELINE):
        self.bundle = joblib.load(path)
        self.labels = self.bundle["labels"]
        self.temperature = self.bundle["temperature"]
        self.seconds = self.bundle["seconds"]
        self.size = path.stat().st_size

    def predict(self, texts, calibrated=True):
        logits = self.bundle["pipeline"].decision_function(texts)
        return probabilities(logits, self.temperature if calibrated else 1)


def main():
    parser = argparse.ArgumentParser(description="Train and calibrate the department baseline.")
    parser.parse_args()
    bundle = fit_baseline(read_tickets("train"), read_tickets("validation"))
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    temp = BASELINE.with_suffix(".tmp")
    joblib.dump(bundle, temp, compress=3)
    temp.replace(BASELINE)
    print(f"Saved baseline to {BASELINE} ({bundle['seconds']:.2f} s)")


if __name__ == "__main__":
    main()
