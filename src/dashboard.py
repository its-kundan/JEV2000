import pandas as pd
import plotly.express as px
import streamlit as st

from src.common import DATA, LABELS, OUT, SEED, device_run, fingerprint, load_json, read
from src.inference import NAMES, InferenceModel, available
from src.router import DEPARTMENTS

COLORS = ["#56d8bf", "#89abff", "#f2b873", "#ed8caa", "#b8a4ff", "#83c9ef", "#c5dc8a", "#e8bfa7"]
EXAMPLES = [
    "I was charged twice and want my money back.",
    "My browser keeps crashing as soon as I upload a PDF.",
    "I upgraded my phone and can no longer get the login codes.",
    "Is the desk mount included with the smaller unit?",
    "When will my parcel arrive? It hasn't moved since Tuesday.",
]


def model_status():
    return {key: available(key) for key in NAMES}


def plot(figure):
    figure.update_layout(template="plotly_dark", paper_bgcolor="#101620", plot_bgcolor="#1a2533",
                         margin=dict(l=18, r=12, t=32, b=22), font_color="#e8eef7")
    st.plotly_chart(figure, width="stretch")


def dataset_ready():
    return (DATA / "provenance.json").is_file() and all((DATA / f"{split}.csv").is_file()
                for split in ("train", "validation", "test"))


def benchmark_data():
    path = OUT / "metrics/benchmark.json"
    if not path.is_file() or not dataset_ready():
        return None
    data = load_json(path)
    if data["data"] != fingerprint("train", "validation", "test"):
        return None
    data["results"] = {mode: row for mode, row in data["results"].items()
                       if available(mode) and row["training_run"] == load_json(OUT / mode / "run.json")}
    return data if data["results"] else None


def overview():
    st.title("System 1 Support Intelligence")
    st.markdown("#### Fine-Tuning & JEPA Lab")
    st.write("Route an incoming ticket to the right team with one model inference. "
             "Send low-confidence predictions to human review.")
    st.subheader("Routing architecture")
    st.markdown("**Ticket → Intent model → Confidence gate → Destination**")
    columns = st.columns(4)
    for column, (name, detail) in zip(columns, [
        ("Ticket", "Customer submits an English support request."),
        ("Intent model", "TF-IDF, fine-tuned DistilBERT, or JEPA probe scores eight intents."),
        ("Confidence gate", "Compare the highest score with the selected threshold."),
        ("Destination", "Send to the mapped department or Human Review Queue."),
    ], strict=True):
        with column:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.write(detail)
    if dataset_ready():
        meta = load_json(DATA / "provenance.json")
        for column, (label, value) in zip(st.columns(3), [
            ("Support tickets", f"{meta['tickets']:,}"), ("Intent classes", 8),
            ("Scenario families", meta["families"]),
        ], strict=True):
            column.metric(label, value)
    st.subheader("Experiment design")
    st.write("System 1 is used here as a description of direct inference. The router predicts an intent "
             "in a single classification pass, with no multi-step reasoning loop.")
    st.markdown(
        "- **TF-IDF + Logistic Regression** measures the performance of a lightweight lexical baseline.\n"
        "- **DistilBERT** tests whether updating a pretrained encoder on labeled tickets helps on unseen families.\n"
        "- **Text JEPA** predicts hidden span embeddings from masked context, then freezes the learned encoder "
        "for linear probing. An identical pretrained encoder without JEPA updates serves as the control."
    )
    st.caption("The brief’s “2000 option” is interpreted as 2,000 tickets. “JEV” is interpreted as Joint-Embedding Predictive Architecture (JEPA).")


def dataset():
    st.title("Dataset Explorer")
    if not dataset_ready():
        st.info("Generate the CSVs with `python3 -m data.generate_dataset`.")
        return
    meta = load_json(DATA / "provenance.json")
    release_test = benchmark_data() is not None
    splits = ["train", "validation"] + (["test"] if release_test else [])
    data = pd.concat([read(split) for split in splits], ignore_index=True)
    st.write(f"{meta['source']} · {meta['tickets']:,} tickets · {meta['families']} families · seed {meta['seed']}")
    st.caption(meta["limitation"])
    st.write("Split sizes · 1,400 train / 300 validation / 300 test")
    counts = data.groupby(["label", "split"]).size().rename("tickets").reset_index()
    distribution, lengths = st.columns(2)
    with distribution:
        st.subheader("Intent distribution")
        plot(px.bar(counts, x="tickets", y="label", color="split", orientation="h", color_discrete_sequence=COLORS))
    with lengths:
        st.subheader("Ticket length")
        plot(px.histogram(data.assign(words=data.text.str.split().str.len()), x="words", color="split",
                          barmode="overlay", opacity=.65, color_discrete_sequence=COLORS))
    if not release_test:
        st.caption("Test examples unlock after final evaluation.")
    split = st.selectbox("Split", splits)
    label = st.selectbox("Intent", ["All intents", *LABELS])
    sample = data[data.split.eq(split)]
    if label != "All intents":
        sample = sample[sample.label.eq(label)]
    st.dataframe(sample[["ticket_id", "label", "text", "family"]].head(30), hide_index=True, width="stretch")


def finetuning():
    st.title("Fine-Tuning Lab")
    st.write("DistilBERT’s six transformer blocks and pretrained embeddings are updated along with a new "
             "eight-class head. The best checkpoint is selected by validation macro F1.")
    path = OUT / "distilbert"
    device = device_run(str, "auto", SEED)
    st.caption(f"Available training device · {device.upper()}")
    if not (path / "run.json").is_file():
        st.info("Run `python3 -m src.finetune_distilbert` from the project root.")
    else:
        run = load_json(path / "run.json")
        first, second, third = st.columns(3)
        first.metric("Selected epoch", run["best_epoch"])
        second.metric("Trainable parameters", f"{run['trainable_parameters']:,}")
        third.metric("Training device", run["device"].upper())
        with st.expander("Saved training configuration"):
            st.json(run["config"])
            st.caption("Inference checkpoint · artifacts/distilbert/model.safetensors")
        if run.get("smoke"):
            st.warning("This checkpoint came from a step-capped smoke run.")
    if (path / "history.json").is_file():
        history = pd.DataFrame(load_json(path / "history.json"))
        left, right = st.columns(2)
        with left:
            plot(px.line(history, x="epoch", y=["train_loss", "validation_loss"],
                         markers=True, labels={"value": "Loss", "variable": "Series"},
                         color_discrete_sequence=COLORS))
        with right:
            plot(px.line(history, x="epoch", y="validation_macro_f1", markers=True,
                         range_y=[0, 1], color_discrete_sequence=COLORS))
        st.dataframe(history, hide_index=True, width="stretch")


def jepa():
    st.title("JEPA Research Lab")
    st.write("A compact text experiment using the first two pretrained DistilBERT blocks by default. "
             "JEPA is a predictive representation-learning objective. The frozen encoder and probe serve direct inference.")
    a, b, c = st.columns(3)
    for column, title, detail in [
        (a, "Context encoder", "Encodes visible content and position-preserving [MASK] placeholders."),
        (b, "EMA target encoder", "Sees the original sequence. Stop-gradient isolates its latent targets."),
        (c, "Predictor", "Uses pooled context and learned target positions to predict the masked states."),
    ]:
        with column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(detail)
    st.code("Original   The parcel has missed two scans .\n"
            "Context    The parcel has [MASK] [MASK] scans .\n"
            "Targets    contextual latents at the two masked positions", language="text")
    st.write("Masked cosine distance aligns predicted and teacher states. The teacher tracks the student with "
             "cosine-scheduled EMA momentum from 0.996 to 0.999. A variance floor on pooled context "
             "and logged feature spread help diagnose representation collapse.")
    folder = OUT / "jepa"
    if not (folder / "history.json").is_file():
        st.info("Run `python3 -m src.train_jepa` and `python3 -m src.linear_probe`.")
        return
    history = pd.DataFrame(load_json(folder / "history.json"))
    left, right = st.columns(2)
    with left:
        plot(px.line(history, x="epoch", y=["loss", "latent_loss"], markers=True,
                     labels={"value": "Loss", "variable": "Series"}, color_discrete_sequence=COLORS))
    with right:
        plot(px.line(history, x="epoch", y=["context_std", "target_std", "prediction_std"], markers=True,
                     labels={"value": "Feature spread", "variable": "Series"}, color_discrete_sequence=COLORS))
    st.dataframe(history, hide_index=True, width="stretch")
    if (folder / "pretrain.json").is_file():
        pretrain = load_json(folder / "pretrain.json")
        with st.expander("Saved pretraining configuration"):
            st.json(pretrain["config"])
            st.caption(f"{pretrain['trainable_parameters']:,} trainable parameters · "
                       "Research checkpoint · artifacts/jepa/research/jepa.safetensors")
    if available("jepa") and (folder / "pca_validation.csv").is_file():
        st.subheader("Validation embeddings")
        st.caption("PCA fitted on unmasked, frozen training embeddings and applied to held-out validation families.")
        coords = pd.read_csv(folder / "pca_validation.csv")
        coords["excerpt"] = coords.text.str.replace("\n", " ").str.slice(0, 100)
        plot(px.scatter(coords, x="pc1", y="pc2", color="label", hover_data=["excerpt"],
                        opacity=.75, color_discrete_sequence=COLORS))
        temperature = load_json(folder / "run.json")["temperature"]
        st.write(f"Probe temperature · **{temperature:.3f}**. Fit on validation by minimizing negative log "
                 "likelihood. Softmax scores summarize model confidence on similar data.")
    else:
        st.info("Train the linear probe to create the embedding view.")


@st.cache_resource(show_spinner=False)
def cached_model(mode, run_metadata, device):
    model = InferenceModel(mode, device)
    model.probabilities(["Support ticket"])
    return model


def live():
    st.title("Live Ticket Router")
    ready = [mode for mode in ("baseline", "distilbert", "jepa") if available(mode)]
    if not ready:
        st.info("No models are ready. Generate the data and train the baseline to enable routing.")
        return
    st.caption("One predicted intent per ticket. Scores estimate confidence and may be unreliable on unfamiliar requests.")
    example = st.selectbox("Load an example", ["Enter my own ticket", *EXAMPLES])
    if example == EXAMPLES[0]:
        st.caption("The brief’s example appears in the default training data. Try custom wording to explore generalization.")
    with st.form(f"ticket-{example}"):
        text = st.text_area("Customer ticket", value="" if example == "Enter my own ticket" else example,
                            height=140, max_chars=12000)
        model_col, threshold_col = st.columns(2)
        mode = model_col.selectbox("Selected model", ready, format_func=NAMES.get)
        threshold = threshold_col.slider("Human review threshold", 0., 1., .55, step=.01)
        compare = st.checkbox("Compare all available classifiers", value=True)
        with st.expander("Department mapping"):
            departments = {label: st.text_input(label, team) for label, team in DEPARTMENTS.items()}
        submitted = st.form_submit_button("Route ticket", type="primary")
    if not submitted:
        return
    if not text.strip() or any(not name.strip() for name in departments.values()):
        st.error("Enter a ticket and a destination for every department.")
        return
    with st.spinner("Scoring ticket…"):
        results = {}
        for key in ready if compare else [mode]:
            signature = (OUT / key / "run.json").read_text()
            results[key] = device_run(
                lambda device: cached_model(key, signature, device).ticket(text, threshold, departments),
                "cpu" if key == "baseline" else "auto", SEED)
    pick = results[mode]
    st.subheader(pick["route"])
    st.caption(pick["model"])
    first, second, third = st.columns(3)
    first.metric("Predicted intent", pick["category"])
    second.metric("Confidence", f"{pick['confidence']:.1%}")
    third.metric("Inference latency", f"{pick['latency_ms']:.1f} ms")
    chart = pd.DataFrame(pick["top"], columns=["Intent", "Score"])
    plot(px.bar(chart, x="Score", y="Intent", orientation="h", range_x=[0, 1],
                color_discrete_sequence=COLORS).update_yaxes(autorange="reversed"))
    if compare:
        table = pd.DataFrame([
            {**{k: v for k, v in item.items() if k != "top"},
             "top_three": ", ".join(f"{label} ({score:.1%})" for label, score in item["top"])}
            for item in results.values()
        ])
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"confidence": st.column_config.NumberColumn(format="percent"),
                                    "latency_ms": st.column_config.NumberColumn(format="%.2f ms")})


def benchmark():
    st.title("Model Benchmark")
    data = benchmark_data()
    if not data:
        st.info("Train the models and run `python3 -m src.evaluation --device cpu` to generate a held-out benchmark.")
        return
    results = data["results"]
    st.caption(f"Held-out set · {data['test_size']} tickets · {data['hardware']['system']}. "
               "Latency includes tokenization or vectorization and prediction. Model loading is timed separately.")
    rows = [{
        "Model": row["model"], "Run": "Smoke" if row["smoke"] else "Full",
        "Accuracy": row["accuracy"], "Macro F1": row["macro_f1"], "Macro precision": row["macro_precision"],
        "Macro recall": row["macro_recall"], "Median (ms)": row["latency_median_ms"],
        "P95 (ms)": row["latency_p95_ms"], "Load (s)": row["load_seconds"], "MiB": row["bytes"] / 2**20,
        "Training (s)": row["training_seconds"], "Trainable params": row["parameters"],
        "Training device": row["training_device"], "Inference device": row["inference_device"],
        "JEPA pretrain (s)": row.get("pretraining_seconds"), "JEPA pretrain params": row.get("pretraining_parameters"),
        "JEPA pretrain device": row.get("pretraining_device"),
    } for row in results.values()]
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        plot(px.bar(table, x=["Accuracy", "Macro F1"], y="Model", orientation="h", barmode="group",
                    range_x=[0, 1], color_discrete_sequence=COLORS))
    with right:
        plot(px.bar(table, x=["Median (ms)", "P95 (ms)"], y="Model", orientation="h", barmode="group",
                    color_discrete_sequence=COLORS))
    mode = st.selectbox("Inspect a model", list(results), format_func=NAMES.get)
    row = results[mode]
    st.caption(f"{row['latency_measurements']} warmed single-ticket timings · "
               f"{row['inference_threads']} PyTorch CPU threads")
    a, b = st.columns(2)
    with a:
        plot(px.imshow(row["confusion"], x=LABELS, y=LABELS, aspect="auto", text_auto=True,
                       labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Teal"))
    with b:
        st.subheader("Per-class results")
        st.dataframe(pd.DataFrame(row["per_class"]).T, width="stretch")
    errors_path = OUT / "metrics" / f"{mode}_errors.csv"
    if errors_path.is_file():
        st.subheader("Misclassified test tickets")
        errors = pd.read_csv(errors_path)
        st.write(f"{len(errors)} incorrect predictions.")
        st.dataframe(errors.sort_values("confidence", ascending=False).head(30), hide_index=True, width="stretch")


def conclusions():
    st.title("Research Conclusions")
    data = benchmark_data()
    if not data:
        st.info("Conclusions will populate after a held-out run. See Model Benchmark for the evaluation command.")
    else:
        rows = data["results"]
        if any(row["smoke"] for row in rows.values()):
            st.warning("This benchmark includes step-capped smoke runs. Treat these metrics as pipeline checks.")
        best = max(rows.values(), key=lambda row: row["macro_f1"])
        quickest = min(rows.values(), key=lambda row: row["latency_median_ms"])
        st.write(f"**Highest macro F1:** {best['model']} ({best['macro_f1']:.1%}).")
        st.write(f"**Fastest median inference:** {quickest['model']} ({quickest['latency_median_ms']:.2f} ms).")
        for mode, control in [("distilbert", "baseline"), ("jepa", "reference")]:
            if mode in rows and control in rows:
                delta = 100 * (rows[mode]["macro_f1"] - rows[control]["macro_f1"])
                st.write(f"**{NAMES[mode]} vs. {NAMES[control]}:** {delta:+.1f} percentage points in macro F1.")
        worst = min(best["per_class"], key=lambda label: best["per_class"][label]["f1-score"])
        st.write(f"The highest-scoring model struggled most on **{worst}**, "
                 f"with F1 of {best['per_class'][worst]['f1-score']:.1%}.")
    st.subheader("Scope and next experiments")
    st.write("The corpus has 80 synthetic scenario families and English-only inputs. Multi-intent tickets use a "
             "single primary label. Each label contributes one or two families to the test set, so estimates "
             "are sensitive to which scenarios are held out.")
    st.write("Validate on de-identified, independently labeled real tickets. Repeat across family splits and seeds, "
             "compare JEPA masking and encoder depth, and measure review coverage and error rate before choosing a threshold.")


PAGES = {"Overview": overview, "Dataset Explorer": dataset, "Fine-Tuning Lab": finetuning,
         "JEPA Research Lab": jepa, "Live Ticket Router": live, "Model Benchmark": benchmark,
         "Research Conclusions": conclusions}
