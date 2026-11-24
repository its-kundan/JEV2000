import streamlit as st

from src.router_dashboard import benchmark, heading

heading("Research Findings", "Observed behavior, vendor context and open questions.")
st.subheader("Structured decisions")
st.markdown("""
TypeSafe describes Jev as a hosted System One model for answering predefined questions with
typed probabilities. TypeSafe reports that Jev uses Reinforcement Learning for Calibrated Decisions
and samples answers in parallel. A fixed output contract makes the results usable as software inputs.
[TypeSafe's announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

The local simulator illustrates the same kinds of decisions with a closed set of departments and destinations,
boolean urgency and review flags, and an ordinal severity score. The classifier supplies department probabilities.
Deterministic rules choose the destination and flag cases for review.
""")
st.subheader("Measured synthetic results")
result = benchmark()
if result:
    for name, model in result["models"].items():
        test = model["splits"]["test"]
        before = test["calibration"]["Before scaling"]["ece"]
        after = test["calibration"]["After scaling"]["ece"]
        st.write(f"**{name}** · {test['report']['macro avg']['f1-score']:.1%} test macro F1 · "
                 f"ECE {before:.3f} before and {after:.3f} after temperature scaling")
    for name in result["models"]:
        rows = [row for row in result["routing"] if row["model"] == name]
        if rows:
            first, last = rows[0], rows[-1]
            st.write(f"**{name} retrieval** · {first['recall_at_k']:.1%} recall@{first['k']} and "
                     f"{last['recall_at_k']:.1%} recall@{last['k']}.")
            if first["recall_at_k"] == last["recall_at_k"]:
                st.caption("A larger shortlist did not recover missed departments in this run.")
            representative = next((row for row in rows if row["k"] == 15), first)
            st.write(f"**{name} routing at K={representative['k']}, threshold "
                     f"{representative['threshold']:.0%}** · {representative['auto_coverage']:.1%} "
                     f"automatic coverage · {representative['misrouting_rate']:.1%} of all tickets misrouted.")
else:
    st.info("Run the benchmark to populate measured findings.")
st.subheader("Limitations and future work")
st.markdown("""
- Synthetic tickets have repeated product and region terms, narrow issue language, and one held-out narrative family per split. Test scores reflect this generator.
- A hard department filter can exclude the correct queue. Evaluate multiple departments or semantic retrieval on human-labeled tickets.
- Calibration describes aggregate behavior on similar data. Measure reliability under changes in language, geography and ticket mix before relying on automated routes.
- Severity and ambiguity heuristics need human labels and operational review data. In the offline flow, the only learned probability used for abstention is department confidence.
""")
