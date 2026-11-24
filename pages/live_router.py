import streamlit as st

from data.generate_destinations import REGIONS
from src.jev_simulator import Simulator
from src.router_config import CONFIG
from src.router_dashboard import flow, get_retriever, heading, select_classifier, show_route
from src.routing_engine import route

heading("Live Support Ticket Router", "Inspect classification, shortlist retrieval, decision and the final routing action.")
retriever = get_retriever()
classifier = select_classifier()
with st.form("router"):
    ticket = st.text_area("Customer ticket", "I was charged twice and I need an urgent refund.", height=150,
                          max_chars=6000)
    a, b, c = st.columns(3)
    region = a.selectbox("Customer region", ["Unspecified", *REGIONS])
    k = b.slider("Top-K destinations", 5, 50, CONFIG["top_k"])
    threshold = c.slider("Confidence threshold", 0., 1., CONFIG["confidence_threshold"], .05)
    st.caption("Tickets and decisions stay on this machine.")
    submit = st.form_submit_button("Route ticket", type="primary")
if submit:
    if not ticket.strip():
        st.error("Enter a customer ticket.")
    else:
        with st.spinner("Routing ticket…"):
            result = route(ticket, classifier, retriever, Simulator(), k, threshold,
                           region=None if region == "Unspecified" else region)
        st.session_state["router_result"] = result
        st.session_state["router_context"] = dict(model=classifier.name, k=k, threshold=threshold)
if "router_result" in st.session_state:
    result = st.session_state.router_result
    context = st.session_state.router_context
    st.caption(f"Last submitted ticket · {context['model']} · threshold {context['threshold']:.0%}")
    flow([("Catalog", f"{retriever.total:,} destinations"), ("Department filter", f"{result.filtered} queues"),
          ("Shortlist", f"{len(result.state.candidates) if result.state else 0} candidates"),
          ("Action", "Auto-route" if result.destination else "Human review")])
    show_route(result)
    st.page_link("pages/jev_inspector.py", label="Inspect the decision", icon=":material/arrow_forward:")
