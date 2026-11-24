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


def test_router_confidence_and_human_escalation():
    from data.generate_destinations import build_destinations
    from src.candidate_retriever import Retriever
    from src.jev_simulator import Simulator
    from src.routing_engine import route

    class Classifier:
        labels = ["Billing", "Refund"]

        def predict(self, texts):
            return [[.85, .15]]

    classifier = Classifier()
    catalog = Retriever(build_destinations())
    backend = Simulator()
    text = "For Atlas Cloud APAC I was charged twice for one order."
    high = route(text, classifier, catalog, backend, threshold=.85)
    assert high.destination == "BILLING-DUPLICATE-CHARGE-APAC-001"
    assert high.state and high.decision and high.latency_ms >= high.retrieval_ms
    low = route(text, classifier, catalog, backend, threshold=.90)
    assert low.destination is None and "below threshold" in low.reasons[0]
    multi = route(text + " I also need to update my password ASAP.", classifier, catalog, backend)
    assert multi.destination is None and multi.decision.urgency
