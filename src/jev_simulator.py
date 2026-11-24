import re
from time import perf_counter

from src.schemas import Decision, questions_for

SIMULATOR = "Local Jev-style Simulator — Not TypeSafe Jev"


class Simulator:
    name = SIMULATOR

    def decide(self, state, questions):
        start = perf_counter()
        if questions != questions_for(state):
            raise ValueError("The simulator expects the router's five typed questions.")
        text = state.ticket_text.casefold()
        urgent = bool(re.search(r"\b(urgent|asap|blocking|outage|down for everyone|immediately)\b", text))
        critical = bool(re.search(r"\b(outage|everyone|entire service)\b", text)) and urgent
        severity = 5 if critical else 4 if urgent else 2
        first = state.candidates[0]
        tied = len(state.candidates) > 1 and first.ranking_score - state.candidates[1].ranking_score < .01
        escalate = tied or first.similarity < .08 or bool(
            re.search(r"\b(i also|not sure|don't know where|speak to (a )?human)\b", text)
        )
        return Decision(state.department, urgent, severity, escalate, first.destination_id, self.name,
                        (perf_counter() - start) * 1000)
