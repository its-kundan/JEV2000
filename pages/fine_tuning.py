import shlex
import subprocess
import sys
from threading import Lock

import pandas as pd
import plotly.express as px
import streamlit as st

from data.prepare_dataset import DATA
from src.common import ROOT, load_json
from src.router_config import DISTILBERT, INITIAL, LOGS
from src.router_dashboard import flow, heading

heading("Model Fine-Tuning Lab", "Supervised eight-department classification with DistilBERT")
flow([("Tokenizer", "Up to 128 tokens"), ("Pretrained encoder", "6 transformer blocks"),
      ("Classification layers", "New head, eight logits"), ("Calibration", "One validation-fitted temperature")])

@st.cache_resource
def training_process():
    return {"lock": Lock(), "process": None}


st.subheader("Training configuration")
with st.form("training"):
    a, b, c, d = st.columns(4)
    epochs = a.number_input("Epochs", 1, 20, 3)
    batch = b.selectbox("Batch size", [8, 16, 32, 64], index=1)
    device = c.selectbox("Device", ["auto", "cpu", "mps", "cuda"])
    freeze = d.slider("Freeze first N blocks", 0, 6, 0)
    defaults = ".cache/distilbert" if (ROOT / ".cache/distilbert" / "model.safetensors").exists() else ""
    snapshot = st.text_input("Pretrained model ID or local snapshot", value=defaults)
    launch = st.form_submit_button("Launch training", type="primary")
args = ["-m", "src.train_distilbert", "--epochs", str(epochs), "--batch-size", str(batch),
        "--device", device, "--freeze-layers", str(freeze)]
if snapshot.strip():
    args += ["--model", snapshot.strip()]
st.code(shlex.join(["python3", *args]), language="bash")
st.caption("AdamW · 2e-5 initial learning rate · 0.01 weight decay · linear warmup/decay · gradient clipping · patience 2")
job = training_process()
log = LOGS / "dashboard-training.log"
if launch:
    with job["lock"]:
        process = job["process"]
        if process and process.poll() is None:
            st.warning("A dashboard training job is already running.")
        elif not (DATA / "train.csv").exists():
            st.warning("Generate the synthetic tickets before training.")
        else:
            LOGS.mkdir(parents=True, exist_ok=True)
            with log.open("a") as output:
                job["process"] = subprocess.Popen([sys.executable, *args], cwd=ROOT,
                                                  stdout=output, stderr=subprocess.STDOUT)
            st.success(f"Training process started (PID {job['process'].pid}).")
if job["process"]:
    status = job["process"].poll()
    st.caption(f"CLI job · {'running' if status is None else f'exit status {status}'}")
    st.button("Refresh training status")
    with st.expander("CLI log"):
        st.code(log.read_text()[-6000:] if log.exists() else "Waiting for output.")
history_path = LOGS / "distilbert_history.json"
if history_path.exists():
    history = pd.DataFrame(load_json(history_path))
    st.subheader("Recorded learning curves")
    left, right = st.columns(2)
    left.plotly_chart(px.line(history, x="epoch", y=["train_loss", "validation_loss"], markers=True))
    right.plotly_chart(px.line(history, x="epoch", y=["validation_macro_f1", "validation_accuracy"], markers=True))
    st.dataframe(history, hide_index=True, width="stretch")
if (DISTILBERT / "run.json").exists():
    record = load_json(DISTILBERT / "run.json")
    st.subheader("Saved checkpoints")
    cols = st.columns(3)
    cols[0].metric("Best validation epoch", record["best_epoch"])
    cols[1].metric("Trainable parameters", f"{record['trainable_parameters']:,}")
    cols[2].metric("Training duration", f"{record['seconds']:.1f} s")
    st.code(f"{INITIAL.relative_to(ROOT)}\n{DISTILBERT.relative_to(ROOT)}")
    with st.expander("Completed run configuration"):
        st.json(record["config"])
else:
    st.info("A completed training run adds the best checkpoint and calibration metadata.")
st.write("The pretrained control has a newly initialized classification head. Fine-tuning updates the head and all unfrozen encoder weights. Freezing at least one block also freezes token and position embeddings.")
