from dataclasses import asdict
from html import escape

import pandas as pd
import streamlit as st

from data.generate_destinations import CATALOG
from src.benchmarking import classifier_paths, data_signature, load_classifier, model_available
from src.candidate_retriever import Retriever
from src.common import load_json
from src.router_config import METRICS


def heading(title, description):
    st.title(title)
    st.caption(description)


def flow(stages):
    nodes = "".join(f'<div class="router-node"><b>{escape(label)}</b><span>{escape(str(detail))}</span></div>'
                    for label, detail in stages)
    st.markdown(f'<div class="router-flow">{nodes}</div>', unsafe_allow_html=True)


def usable_models():
    return {name: path for name, path in classifier_paths().items() if model_available(path)}


def select_classifier(label="Department classifier"):
    models = usable_models()
    if not models:
        st.info("Train the baseline with `python3 -m src.baseline_model` to enable routing.")
        st.stop()
    choice = st.selectbox(label, list(models), index=len(models) - 1)
    return cached_classifier(models[choice], version(models[choice]))


def version(path):
    artifact = path / "run.json" if path.is_dir() else path
    return artifact.stat().st_mtime_ns


@st.cache_resource
def cached_classifier(path, artifact_version):
    return load_classifier(path)


@st.cache_resource
def cached_retriever(catalog_version):
    return Retriever()


def get_retriever():
    if not CATALOG.exists():
        st.info("Generate the catalog with `python3 -m data.generate_destinations`.")
        st.stop()
    return cached_retriever(version(CATALOG))


def benchmark():
    path = METRICS / "benchmark.json"
    if path.exists():
        result = load_json(path)
        try:
            if result["dataset"] == data_signature():
                return result
            st.warning("Generated data changed after the benchmark. Rerun the evaluation.")
        except FileNotFoundError:
            st.info("Generate the benchmark dataset to view saved results.")
    return None


def show_results(rows):
    frame = pd.DataFrame(rows)
    percent = ("recall_at_k", "proposed_accuracy", "decision_accuracy_when_retrieved",
               "end_to_end_accuracy", "auto_coverage", "auto_accuracy", "human_review",
               "misrouting_rate", "review_label_recall")
    columns = {name: st.column_config.NumberColumn(name.replace("_", " ").title(), format="percent")
               for name in percent}
    st.dataframe(frame, column_config=columns, hide_index=True, width="stretch")


def show_route(result):
    state, decision = result.state, result.decision
    department = max(result.department_probabilities, key=result.department_probabilities.get)
    metrics = st.columns(4)
    metrics[0].metric("Predicted department", department)
    metrics[1].metric("Department confidence", f"{result.department_probabilities[department]:.1%}")
    metrics[2].metric("Shortlisted queues", len(state.candidates) if state else 0)
    metrics[3].metric("End-to-end latency", f"{result.latency_ms:.1f} ms")
    st.caption(f"{result.attempted_backend} · retrieval {result.retrieval_ms:.1f} ms")
    if result.destination:
        st.success(f"Auto-route → {result.destination}")
    else:
        st.warning("Human review · " + " ".join(result.reasons))
    if decision:
        st.write(f"Proposed destination **{decision.destination_id}** · Urgent **{decision.urgency}** · "
                 f"Severity **{decision.severity}/5** · Escalation flag **{decision.escalate}**")
    if state:
        st.dataframe([asdict(candidate) for candidate in state.candidates], hide_index=True, width="stretch")
