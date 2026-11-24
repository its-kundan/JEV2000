import pandas as pd
import plotly.express as px
import streamlit as st

from data.prepare_dataset import DATA, read_tickets
from src.benchmarking import evaluate_routes, sample_tickets
from src.jev_simulator import Simulator
from src.router_config import CONFIG
from src.router_dashboard import benchmark, get_retriever, heading, select_classifier, show_results

heading("Routing Experiments", "Measure which queues survive retrieval and how confidence affects automated coverage.")
result = benchmark()
if result and result["routing"]:
    st.subheader("Candidate recall by shortlist size")
    frame = pd.DataFrame(result["routing"])
    figure = px.line(frame, x="k", y="recall_at_k", color="model", markers=True,
                     labels={"k": "Top-K queues", "recall_at_k": "Candidate recall", "model": "Classifier"})
    figure.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(figure, width="stretch")
    show_results(result["routing"])
    if result["latency"]:
        st.subheader("Measured end-to-end latency")
        st.dataframe(pd.DataFrame(result["latency"])[["model", "tickets", "latency_ms", "p95_ms",
                                                      "backend_latency_ms"]], hide_index=True)
st.caption("Candidate recall counts tickets whose gold ID survives retrieval. Conditional decision accuracy uses those tickets as the denominator. End-to-end accuracy counts correct automatic routes over all tickets, including abstentions.")
st.subheader("Run a routing policy")
if not (DATA / "test.csv").exists():
    st.info("Generate the ticket partitions before running an experiment.")
    st.stop()
retriever = get_retriever()
model = select_classifier()
simulator = Simulator()
with st.form("experiment"):
    a, b, c = st.columns(3)
    k = a.slider("Top-K candidates", 5, 50, CONFIG["top_k"])
    threshold = b.slider("Confidence threshold", 0., 1., CONFIG["confidence_threshold"], .05)
    tickets = c.selectbox("Seeded test sample", [10, 25, 50, 100, 250], index=1)
    sweep = st.checkbox("Compare thresholds 0.60, 0.70, 0.80, 0.90, 0.95")
    submit = st.form_submit_button("Run experiment", type="primary")
if submit:
    examples = sample_tickets(read_tickets("test"), tickets)
    thresholds = sorted({threshold, .60, .70, .80, .90, .95}) if sweep else [threshold]
    with st.spinner(f"Evaluating {len(examples)} synthetic tickets…"):
        rows = [dict(model=model.name, **evaluate_routes(model, examples, retriever, simulator, k, value))
                for value in thresholds]
    st.session_state["routing_experiment"] = rows
if "routing_experiment" in st.session_state:
    rows = st.session_state.routing_experiment
    show_results(rows)
    if len(rows) > 1:
        chart = px.line(pd.DataFrame(rows), x="threshold", y=["auto_coverage", "auto_accuracy", "human_review"],
                        markers=True, labels={"value": "Share of tickets", "threshold": "Confidence threshold"})
        chart.update_yaxes(tickformat=".0%")
        st.plotly_chart(chart, width="stretch")
