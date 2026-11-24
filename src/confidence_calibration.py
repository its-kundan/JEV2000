import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def fit_temperature(logits, labels):
    indices = np.arange(len(labels))
    result = minimize_scalar(
        lambda log_t: -log_softmax(logits / np.exp(log_t), axis=1)[indices, labels].mean(),
        bounds=(-3, 3), method="bounded",
    )
    return float(np.exp(result.x))


def probabilities(logits, temperature=1.0):
    return softmax(np.asarray(logits) / temperature, axis=-1)


def calibration_metrics(probs, labels, bins=10):
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    memberships = np.minimum((confidence * bins).astype(int), bins - 1)
    reliability = []
    for index in range(bins):
        mask = memberships == index
        if mask.any():
            reliability.append(dict(count=int(mask.sum()), confidence=float(confidence[mask].mean()),
                                    accuracy=float(correct[mask].mean())))
    ece = sum(row["count"] * abs(row["confidence"] - row["accuracy"]) for row in reliability) / len(labels)
    brier = ((probs - np.eye(probs.shape[1])[labels]) ** 2).sum(axis=1).mean()
    return dict(ece=float(ece), brier=float(brier), reliability=reliability)
