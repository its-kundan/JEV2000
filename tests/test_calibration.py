import numpy as np

from src.confidence_calibration import calibration_metrics, fit_temperature, probabilities


def test_temperature_preserves_predictions_and_reduces_overconfidence():
    logits = np.array([[8., 0.], [0., 8.], [8., 0.], [0., 8.]])
    labels = np.array([0, 1, 1, 1])
    before = probabilities(logits)
    after = probabilities(logits, fit_temperature(logits, labels))
    np.testing.assert_array_equal(before.argmax(axis=1), after.argmax(axis=1))
    np.testing.assert_allclose(after.sum(axis=1), 1)
    assert calibration_metrics(after, labels)["ece"] < calibration_metrics(before, labels)["ece"]
    assert calibration_metrics(after, labels)["brier"] < calibration_metrics(before, labels)["brier"]
