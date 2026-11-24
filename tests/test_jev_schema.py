from dataclasses import replace

import pytest

from src.jev_simulator import SIMULATOR, Simulator
from src.schemas import Candidate, DecisionState, questions_for


@pytest.fixture
def state():
    return DecisionState("Urgent: payment did not go through.", "Billing", .91,
                         (Candidate("QUEUE-1", "Billing", "Payment failure", "Payments", "APAC",
                                    "Atlas Cloud", .9, .9),), "APAC")


def test_simulator_returns_a_valid_local_decision(state):
    questions = questions_for(state)
    assert set(questions["destination"]["criteria"]) == {"QUEUE-1"}
    decision = Simulator().decide(state, questions).validate(state)
    assert decision.source == SIMULATOR
    assert decision.urgency is True
    assert decision.severity == 4
    assert decision.destination_id == "QUEUE-1"


def test_invalid_questions_and_typed_values(state):
    with pytest.raises(ValueError):
        Simulator().decide(state, {})
    local = Simulator().decide(state, questions_for(state))
    with pytest.raises(ValueError):
        replace(local, severity=True).validate(state)
    with pytest.raises(ValueError):
        replace(local, destination_id="outside-shortlist").validate(state)
