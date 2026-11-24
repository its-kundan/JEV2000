"""Typed boundary shared by the local and hosted decision backends."""
from dataclasses import dataclass, field
from math import isfinite

from data.generate_destinations import REGIONS, TAXONOMY


@dataclass(frozen=True)
class Candidate:
    destination_id: str
    department: str
    subcategory: str
    description: str
    region: str
    product: str
    similarity: float
    ranking_score: float


@dataclass(frozen=True)
class DecisionState:
    ticket_text: str
    department: str
    department_confidence: float
    candidates: tuple[Candidate, ...]
    region: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.ticket_text.strip() or self.department not in TAXONOMY:
            raise ValueError("A ticket and valid predicted department are required.")
        if self.region is not None and self.region not in REGIONS:
            raise ValueError("Unknown region.")
        ids = {candidate.destination_id for candidate in self.candidates}
        if len(ids) != len(self.candidates) or not 1 <= len(ids) <= 50:
            raise ValueError("Expected one to 50 unique shortlisted destinations.")
        probability(self.department_confidence)


def probability(value):
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Expected a numeric probability in [0, 1].")
    return float(value)


SEVERITY = ("Informational request", "Minor inconvenience", "Material disruption",
            "Major workflow blocked", "Critical and widespread impact")


def questions_for(state):
    return {
        "department": {"type": "choice", "instructions": "Identify the primary requested support action.",
                       "criteria": {name: ", ".join(issues) for name, issues in TAXONOMY.items()}},
        "urgency": {"type": "noul", "instructions": "Does this ticket need urgent attention?"},
        "severity": {"type": "score", "instructions": "Assess the operational impact on the customer.",
                     "criteria": list(SEVERITY)},
        "escalate": {"type": "noul", "instructions": "Does this need human review because it is ambiguous, multi-intent, or has no suitable queue?"},
        "destination": {"type": "choice", "instructions": "Choose the best queue for the primary requested action. Consider issue, product and region.",
                        "criteria": {c.destination_id: c.description for c in state.candidates}},
    }


@dataclass(frozen=True)
class Decision:
    department: str
    urgency: bool
    severity: int
    escalate: bool
    destination_id: str
    source: str
    latency_ms: float

    def validate(self, state):
        if self.department not in TAXONOMY:
            raise ValueError("Unknown decision department.")
        if type(self.urgency) is not bool or type(self.escalate) is not bool:
            raise ValueError("Urgency and escalation must be booleans.")
        if type(self.severity) is not int or not 1 <= self.severity <= 5:
            raise ValueError("Severity must be an integer in [1, 5].")
        if self.destination_id not in {candidate.destination_id for candidate in state.candidates}:
            raise ValueError("Selected destination is outside the shortlist.")
        return self
