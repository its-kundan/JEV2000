import pandas as pd
import streamlit as st

from src.router_dashboard import benchmark, flow, usable_models

st.markdown("""
<div class="router-hero">
<div class="router-kicker">CUSTOMER SUPPORT ROUTING</div>
<h1>System One Intelligent<br>Support Router</h1>
<p>Classify a support ticket, retrieve the most relevant queues, and make a typed decision across 2,000 possible destinations.</p>
</div>
""", unsafe_allow_html=True)
for column, label, value in zip(st.columns(3), ("Configured destinations", "Departments", "Max shortlist"),
                                ("2,000", 8, 50)):
    column.metric(label, value)
st.subheader("Routing architecture")
flow([("Customer ticket", "Text and optional region"), ("Department model", "TF-IDF or DistilBERT"),
      ("Queue retrieval", "Cosine similarity and context"), ("Typed decision", "Local rules and typed outputs"),
      ("Routing action", "Destination or human review")])
st.markdown("""
TypeSafe describes **System One** as a model class for structured decisions inside software.
Its hosted model **Jev** answers predefined questions with typed values and probabilities.
This demo lets application code validate those answers and apply a review policy.
[Read the TypeSafe announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
""")
st.write("**Available routing classifiers** · " + (", ".join(usable_models()) or "Generate data and train the baseline"))
st.subheader("Measured experiments")
results = benchmark()
if results:
    st.dataframe(pd.DataFrame([
        {"Model": item["name"], "Test accuracy": f"{item['splits']['test']['report']['accuracy']:.1%}",
         "Test macro F1": f"{item['splits']['test']['report']['macro avg']['f1-score']:.1%}",
         "Median inference": f"{item['splits']['test']['latency_ms']:.2f} ms"}
        for item in results["models"].values()
    ]), hide_index=True, width="stretch")
    st.caption("Measured locally on held-out synthetic tickets.")
else:
    st.info("Run `python3 -m src.benchmarking` to generate measured results.")
st.page_link("pages/live_router.py", label="Try a live ticket", icon=":material/arrow_forward:")
