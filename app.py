import streamlit as st

from src.dashboard import PAGES, model_status
from src.inference import NAMES

st.set_page_config(page_title="System 1 Support Intelligence", page_icon="◈", layout="wide")
with st.sidebar:
    st.markdown("### System 1 Support Intelligence")
    st.caption("Fine-Tuning & JEPA Lab")
    page = st.radio("Workspace", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("LOCAL CHECKPOINTS")
    for key, ready in model_status().items():
        st.markdown(f"{'●' if ready else '○'} {NAMES[key]}")
PAGES[page]()
