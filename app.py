import streamlit as st

st.set_page_config(page_title="System One · Support Router", page_icon="◈", layout="wide")

st.markdown("""
<style>
[data-testid="stMainBlockContainer"] {max-width: 1320px; padding-top: 2.4rem;}
[data-testid="stSidebar"] {border-right: 1px solid #243147;}
[data-testid="stSidebarNavLink"][aria-current="page"] {background: #173433; color: #6ce5c7;}
[data-testid="stMetric"] {background: #101c2c; border: 1px solid #293b50; border-radius: 12px; padding: 1rem;}
[data-testid="stMetricValue"] {font-size: clamp(1rem, 2.7vw, 2rem);}
[data-testid="stForm"] {border-color: #293b50;}
h1,h2,h3 {letter-spacing: -.025em;}
.router-hero {background: radial-gradient(ellipse at 92% 5%, #203f51 0%, transparent 52%),
  linear-gradient(135deg, #142539, #101b2a); border: 1px solid #314864; border-radius: 18px;
  padding: clamp(1.3rem, 4vw, 3rem); margin-bottom: 1.2rem;}
.router-hero h1 {line-height: 1.08; font-size: clamp(2rem, 5vw, 4rem); margin: .35em 0 .3em; padding: 0;}
.router-hero p {color: #b7c8dd; max-width: 60ch; margin-bottom: 0;}
.router-kicker {color:#73e2c8; font: 600 .78rem ui-monospace, monospace; letter-spacing:.13em;}
.router-flow {display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: .55rem;}
.router-node {border: 1px solid #314564; background: #101c2c; border-radius: 10px; padding: .9rem;}
.router-node span {display: block; color: #aebfd4; font-size: .86rem;}
</style>
""", unsafe_allow_html=True)

pages = [
    st.Page("pages/overview.py", title="Executive Overview", icon=":material/dashboard:", default=True),
    st.Page("pages/dataset_explorer.py", title="Dataset & Destinations", icon=":material/database:"),
    st.Page("pages/fine_tuning.py", title="Model Fine-Tuning Lab", icon=":material/science:"),
    st.Page("pages/live_router.py", title="Live Ticket Router", icon=":material/route:"),
    st.Page("pages/jev_inspector.py", title="Decision Inspector", icon=":material/data_object:"),
    st.Page("pages/model_evaluation.py", title="Model Evaluation", icon=":material/monitoring:"),
    st.Page("pages/routing_experiments.py", title="Routing Experiments", icon=":material/tune:"),
    st.Page("pages/research_findings.py", title="Research Findings", icon=":material/article:"),
]
navigation = st.navigation(pages)
with st.sidebar:
    st.markdown("**◈ SYSTEM ONE / SUPPORT ROUTER**")
    st.caption("2,000 synthetic destinations")
navigation.run()
