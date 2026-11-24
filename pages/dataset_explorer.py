import pandas as pd
import plotly.express as px
import streamlit as st

from data.generate_destinations import CATALOG, TAXONOMY
from data.prepare_dataset import DATA, SPLITS, read_tickets
from src.common import load_json
from src.router_dashboard import heading

heading("Dataset & Destination Explorer", "Synthetic catalog and family-held-out ticket dataset")
if CATALOG.exists():
    catalog = pd.DataFrame(load_json(CATALOG))
    a, b, c = st.columns(3)
    a.metric("Destinations", f"{len(catalog):,}")
    b.metric("Departments", catalog.department.nunique())
    c.metric("Issue categories", catalog.subcategory.nunique())
    st.subheader("Destination hierarchy")
    st.plotly_chart(px.treemap(catalog, path=["department", "subcategory", "region"]).update_layout(height=430),
                    width="stretch")
    department = st.selectbox("Filter department", ["All departments", *TAXONOMY])
    query = st.text_input("Search by ID, issue, region, or product")
    matches = catalog if department == "All departments" else catalog[catalog.department == department]
    if query.strip():
        search = matches[["destination_id", "subcategory", "description", "team_name"]].agg(" ".join, axis=1)
        matches = matches[search.str.contains(query.strip(), case=False, regex=False)]
    st.caption(f"{len(matches):,} matching synthetic queues")
    st.dataframe(matches, hide_index=True, width="stretch")
if (DATA / "summary.json").exists():
    summary = load_json(DATA / "summary.json")
    st.subheader("Ticket partitions")
    cols = st.columns(3)
    for col, name in zip(cols, SPLITS):
        col.metric(name.title(), f"{summary['partitions'][name]:,}")
    split = st.selectbox("Inspect split", SPLITS)
    tickets = read_tickets(split)
    counts = tickets.department.value_counts().rename_axis("Department").reset_index(name="Tickets")
    st.plotly_chart(px.bar(counts, x="Tickets", y="Department", orientation="h"), width="stretch")
    st.dataframe(tickets.head(100), hide_index=True, width="stretch")
else:
    st.info("Generate data with `python3 -m data.generate_destinations` and `python3 -m data.generate_tickets`.")
st.caption("Narrative templates and their paraphrase variants stay within a single split. Product names and regions are intentionally shared, as they would be in a configured routing catalog. Synthetic accuracy does not establish real-world performance.")
