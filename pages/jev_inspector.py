from dataclasses import asdict

import streamlit as st

from src.router_dashboard import heading
from src.schemas import questions_for

heading("Decision Inspector", "See the actual local decision and the evidence used to route the ticket.")
result = st.session_state.get("router_result")
if not result or not result.state:
    st.info("Submit a ticket in Live Ticket Router to inspect its decision.")
    st.page_link("pages/live_router.py", label="Open Live Ticket Router")
    st.stop()

left, right = st.columns(2)
with left.expander("Structured state", expanded=True):
    st.json(asdict(result.state), expanded=1)
with right.expander("Typed questions", expanded=True):
    st.json(questions_for(result.state), expanded=2)
st.caption("Choice selects from a closed set. The local rules return yes/no flags and a severity level from 1 to 5.")

if result.decision:
    st.success(f"Schema validation passed · {result.decision.source}")
    with st.expander("Validated local decision", expanded=True):
        st.json(asdict(result.decision))
elif result.schema_failure:
    st.error("Schema validation failed. The ticket was sent for human review.")

st.write(f"**Selected routing action** · {result.destination or 'Human review'}")
for reason in result.reasons:
    st.write(reason)
with st.expander("Department probabilities"):
    st.caption("These probabilities come from the local classifier. Destination selection uses retrieval ranks and rules.")
    st.json(result.department_probabilities)
