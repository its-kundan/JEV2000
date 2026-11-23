import pytest

from src.common import LABELS
from src.router import DEPARTMENTS, destination


def test_threshold_and_mapping():
    assert set(DEPARTMENTS) == set(LABELS)
    assert destination("Refund", .65, .65) == "Refund Operations"
    assert destination("Refund", .64, .65) == "Human Review Queue"
    assert destination("Refund", .7, .6, {"Refund": "Custom"}) == "Custom"
    with pytest.raises(ValueError):
        destination("Refund", float("nan"), .6)
