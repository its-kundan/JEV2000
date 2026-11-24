import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.router_dashboard import benchmark, heading

heading("Model Evaluation", "Department performance, latency and confidence on common synthetic splits.")
result = benchmark()
if not result or not result["models"]:
    st.info("Run `python3 -m src.benchmarking` to generate evaluation metrics.")
    st.stop()
split = st.radio("Evaluation split", ["test", "validation"], horizontal=True)
models = result["models"]
rows = []
for model in models.values():
    metrics = model["splits"][split]
    macro = metrics["report"]["macro avg"]
    rows.append({"Model": model["name"], "Accuracy": metrics["report"]["accuracy"], "Macro F1": macro["f1-score"],
                 "Macro precision": macro["precision"], "Macro recall": macro["recall"],
                 "Inference p50 (ms)": metrics["latency_ms"], "Inference p95 (ms)": metrics["p95_ms"],
                 "Checkpoint (MiB)": model["size_bytes"] / 2**20, "Training (s)": model["training_seconds"]})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
             column_config={name: st.column_config.NumberColumn(format="percent") for name in
                            ("Accuracy", "Macro F1", "Macro precision", "Macro recall")})
name = st.selectbox("Inspect model", list(models))
details = models[name]["splits"][split]
report = details["report"]
st.subheader("Per-department performance")
st.dataframe(pd.DataFrame({label: report[label] for label in details["labels"]}).T, width="stretch")
left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    heatmap = px.imshow(details["confusion"], x=details["labels"], y=details["labels"],
                        labels={"x": "Predicted", "y": "Actual"}, text_auto=True, color_continuous_scale="Teal")
    st.plotly_chart(heatmap, width="stretch")
with right:
    st.subheader("Reliability diagram")
    figure = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal calibration",
                                 line=dict(dash="dot", color="#8499b1")))
    scores = []
    for label, data in details["calibration"].items():
        points = data["reliability"]
        figure.add_scatter(x=[p["confidence"] for p in points], y=[p["accuracy"] for p in points],
                           customdata=[p["count"] for p in points], mode="lines+markers", name=label,
                           hovertemplate="Confidence %{x:.1%}<br>Accuracy %{y:.1%}<br>Examples %{customdata}")
        scores.append({"Probabilities": label, "ECE": data["ece"], "Multiclass Brier": data["brier"]})
    figure.update_layout(xaxis_title="Mean confidence", yaxis_title="Observed accuracy",
                         xaxis_range=[0, 1], yaxis_range=[0, 1], legend=dict(orientation="h", y=-.24))
    st.plotly_chart(figure, width="stretch")
    st.dataframe(pd.DataFrame(scores), hide_index=True)
st.caption("Temperature is fit on validation. The initialized-head control uses temperature 1. Inference timings are warmed single-ticket calls on CPU.")
st.subheader("Error analysis")
if details["errors"]:
    st.dataframe(pd.DataFrame(details["errors"]), hide_index=True, width="stretch")
else:
    st.caption("No department errors in this evaluated split.")
