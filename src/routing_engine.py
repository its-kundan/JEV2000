from dataclasses import dataclass
from time import perf_counter

from src.schemas import Decision, DecisionState, questions_for


@dataclass(frozen=True)
class Route:
    state: DecisionState | None
    decision: Decision | None
    department_probabilities: dict
    filtered: int
    retrieval_ms: float
    latency_ms: float
    reasons: tuple[str, ...]
    attempted_backend: str
    schema_failure: bool = False

    @property
    def destination(self):
        return self.decision.destination_id if self.decision and not self.reasons else None


def review_reasons(state, decision, threshold):
    reasons = []
    if state.department_confidence < threshold:
        reasons.append("Department confidence below threshold.")
    if decision.department != state.department:
        reasons.append("Department disagreement with the retrieval filter.")
    if decision.escalate:
        reasons.append("Decision backend flagged the ticket for review.")
    return tuple(reasons)


def route(text, classifier, retriever, backend, k=15, threshold=.70, region=None, metadata=None, prediction=None):
    if not text.strip() or not 0 <= threshold <= 1:
        raise ValueError("Enter a ticket and a threshold in [0, 1].")
    start = perf_counter()
    scores = classifier.predict([text])[0] if prediction is None else prediction
    pairs = dict(zip(classifier.labels, map(float, scores), strict=True))
    department = max(pairs, key=pairs.get)
    candidates, filtered, retrieval_ms = retriever.retrieve(text, department, k, region)
    state = decision = None
    schema_failure = False
    if not candidates:
        reasons = ("No active candidates.",)
    else:
        state = DecisionState(text, department, pairs[department], candidates, region, metadata or {})
        try:
            decision = backend.decide(state, questions_for(state)).validate(state)
        except (ValueError, TypeError, KeyError, AttributeError):
            schema_failure = True
            reasons = ("Decision failed typed schema validation.",)
        else:
            reasons = review_reasons(state, decision, threshold)
    return Route(state, decision, pairs, filtered, retrieval_ms, (perf_counter() - start) * 1000,
                 reasons, backend.name, schema_failure)
