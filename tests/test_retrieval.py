import pandas as pd

from data.generate_destinations import build_destinations
from src.benchmarking import evaluate_routes
from src.candidate_retriever import Retriever
from src.jev_simulator import Simulator


class BillingClassifier:
    labels = ["Billing", "Refund"]

    def predict(self, texts):
        return [[.95, .05] for _ in texts]


def test_candidates_and_recall():
    retriever = Retriever(build_destinations())
    text = "Atlas Cloud APAC: charged twice for one order"
    candidates, filtered, duration = retriever.retrieve(text, "Billing", 5, "APAC")
    assert retriever.total == 2000
    assert len(candidates) == 5 and filtered == 250 and duration >= 0
    assert candidates[0].destination_id == "BILLING-DUPLICATE-CHARGE-APAC-001"
    frame = pd.DataFrame([
        dict(ticket_text=text, region="APAC", destination_id=candidates[0].destination_id,
             requires_human_review=False),
        dict(ticket_text=text, region="APAC", destination_id="REFUND-REFUND-REQUEST-APAC-001",
             requires_human_review=True),
    ])
    metrics = evaluate_routes(BillingClassifier(), frame, retriever, Simulator(), 5, .7)
    assert metrics["recall_at_k"] == .5
    assert metrics["decision_accuracy_when_retrieved"] == 1.
    assert metrics["end_to_end_accuracy"] == .5
    assert metrics["schema_failures"] == 0
