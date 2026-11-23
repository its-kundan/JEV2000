import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

from src.common import OUT, SEED, fingerprint, read, write_json


def main():
    train, val = read("train"), read("validation")
    started = time.perf_counter()
    trials = []
    best = None
    for analyzer, ngrams in [("word", (1, 2)), ("char_wb", (3, 5))]:
        for c in (.5, 2, 8):
            model = make_pipeline(TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, min_df=2),
                                  LogisticRegression(C=c, max_iter=500, random_state=SEED))
            model.fit(train.text, train.label)
            score = f1_score(val.label, model.predict(val.text), average="macro")
            trials.append(dict(analyzer=analyzer, C=c, validation_macro_f1=score))
            if best is None or score > best[0]:
                best = score, model, trials[-1]
    folder = OUT / "baseline"
    folder.mkdir(parents=True, exist_ok=True)
    _, model, selection = best
    head = model[-1]
    joblib.dump(model, folder / "pipeline.joblib", compress=3)
    write_json(folder / "run.json", dict(selection=selection, trials=trials,
               trainable_parameters=head.coef_.size + head.intercept_.size,
               seconds=time.perf_counter() - started, device="cpu", data=fingerprint("train", "validation")))
    print(selection)


if __name__ == "__main__":
    main()
