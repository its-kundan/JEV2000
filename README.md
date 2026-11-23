# System 1 Support Intelligence — Fine-Tuning & JEPA Lab

A local research POC that routes English support tickets to eight departments or a human review queue. “System 1” describes the direct, single-pass inference style. JEPA describes a predictive representation-learning objective. The brief’s “2000 option” is interpreted as 2,000 tickets, and “JEV” as **Joint-Embedding Predictive Architecture**.

## Run on macOS

Use Python 3.11 or later. The recorded run used Python 3.12 on arm64 macOS with PyTorch 2.15.0, Transformers 4.57.6 and scikit-learn 1.9.1. Install inside a virtual environment from the project root.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 -m data.generate_dataset
python3 -m src.baseline
python3 -m src.finetune_distilbert
python3 -m src.train_jepa
python3 -m src.linear_probe
python3 -m src.evaluation --device cpu
python3 -m pytest -q
python3 -m streamlit run app.py --server.headless true --server.address 127.0.0.1
```

Open <http://127.0.0.1:8501>. The two neural training commands require the public DistilBERT checkpoint once. They accept `--model /path/to/local/snapshot` for offline training with an existing copy. Inference reads local artifacts only. Each neural CLI has `--help`, `--device {auto,cpu,mps,cuda}`, and `--max-steps N` for explicitly marked smoke runs. Auto prefers CUDA, then MPS, then CPU. Unavailable devices and supported fallback errors are surfaced in the CLI, and unsupported accelerator operations restart the job on CPU. CPU execution was verified here. MPS and CUDA paths require compatible hardware to verify. Seeds control initialization, ordering, and masking. Numeric results and timing can vary by device and library version.

`data/generated/` contains split CSVs and provenance. `artifacts/` contains local Safetensors checkpoints, Python joblib pipelines, training histories, validation PCA coordinates, and final benchmark/error files. Load joblib artifacts created by this project or from a trusted source. Models and generated data are excluded from Git. Allow roughly 1.2 GB for trained artifacts and the original checkpoint, in addition to the Python environment. The dashboard identifies available checkpoints and can start before training. The benchmark checks data fingerprints and training metadata before showing saved results.

## Data and evaluation protocol

The default benchmark uses 80 authored scenario families across eight intents, with 25 surface variants per family. It includes short requests, longer explanations, informal spelling, frustrated follow-ups, product contexts, and secondary intents. The primary requested action defines the label. The 250 examples per class are split by family before training. Each class has 175 training rows, and 25 or 50 validation/test rows, for exact overall sizes of 1,400 / 300 / 300. Validation chooses hyperparameters, checkpoints and probe temperature. The evaluation CLI is the first stage that reads the test split. The Dataset Explorer reveals test examples after evaluation.

The Bitext corpus overlaps with five of the eight routing categories. To download and preprocess those intents for a separate taxonomy audit, use `python3 -m data.prepare_data --revision main` or supply an immutable dataset revision. It maps selected intents, drops normalized duplicates and conflicting texts, and writes a separate CSV and provenance record under `data/generated/bitext_overlap/`. The verified download yielded 8,134 mapped tickets. The default training and evaluation commands use the synthetic CSVs exclusively. Bitext is itself generated/hybrid data. See the [Bitext dataset card and license](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).

## What trains and what freezes

| Experiment | Trainable components | Selection and scoring |
| --- | --- | --- |
| TF-IDF + Logistic Regression | Fit vocabulary/IDF and logistic coefficients on training tickets | Select word vs. character n-grams and regularization by validation macro F1 |
| Fine-tuned DistilBERT | Pretrained embeddings and all six transformer blocks, plus newly initialized classification layers | AdamW, 2e-5 starting LR, warm-up and linear decay, weight decay, clipping, up to three epochs by default. Save the best validation macro F1 checkpoint with its tokenizer |
| Text JEPA + linear probe | First two DistilBERT blocks, target-position embeddings and MLP predictor during pretraining. Freeze token and position embeddings in the encoder. During probing, freeze the entire context encoder and fit only a standardized logistic head | Pretrain without labels, then use validation macro F1 to choose probe regularization and validation log loss to fit a scalar temperature |
| Pretrained encoder + linear probe | Standardized logistic head on the same two-layer encoder saved before JEPA updates | Same pooling, training labels, regularization grid, and calibration as the JEPA probe |

DistilBERT initialization uses the pinned public checkpoint. See the [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased).

The JEPA student replaces a contiguous span of about 30% of ordinary tokens with mask tokens and averages contextual states. The predictor combines the pooled context with a learned embedding for each selected target position. The EMA teacher sees the original full sequence, and its contextual states at those positions are detached. The loss is masked cosine distance plus a small pooled-context variance penalty. There is no vocabulary reconstruction objective. EMA updates the teacher after each optimizer step, with momentum increasing on a cosine schedule from 0.996 to 0.999. The history tracks context, target and prediction feature spread. Stable spread can identify gross collapse, while the frozen probe tests whether features preserve routing information. This compact, pooled text adaptation uses a single contiguous span and unmasked downstream mean pooling. See the [I-JEPA paper](https://arxiv.org/abs/2301.08243) for the visual architecture that inspired it.

Probe temperature scaling adjusts score sharpness and preserves the predicted class. Baseline and DistilBERT output their native model probabilities. All scores can be unreliable on unfamiliar input. The threshold defaults to 0.55 and is adjustable for the demo. The router compares the best score with the threshold, sending tickets below the threshold to Human Review Queue. Department names are editable in the form.

## Recorded held-out run

All results below came from the same 300 held-out synthetic tickets on CPU. The benchmark UI reads the actual JSON and shows per-class precision, recall and F1, confusion matrices, test errors, training time, parameter counts, device, and artifact sizes. The latency protocol warms up each classifier and times three passes over 24 fixed test tickets, including single-ticket tokenization or vectorization. Model loading is timed separately. Reported artifact sizes cover inference bundles, excluding JEPA’s research-only teacher and predictor checkpoint. JEPA time and trainable parameter counts are broken out by pretraining and probing.

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| TF-IDF + Logistic Regression | 47.3% | 49.0% |
| Fine-tuned DistilBERT | 83.3% | 72.9% |
| Text JEPA + linear probe | 57.0% | 51.8% |
| Matched pretrained encoder + linear probe | 73.0% | 69.8% |

DistilBERT gained 24.0 percentage points of macro F1 over TF-IDF in this run. JEPA lost 18.0 points against its matched pretrained control. The representation objective can improve while transfer degrades. The experiment measures one seed and one synthetic split with one or two test families per class. It does not estimate performance on real customer tickets.

## Manager walkthrough · about 9 minutes

| Time | Page and talking points |
| --- | --- |
| 0:00–1:00 | **Overview.** Explain one-pass intent scoring, the confidence gate, and department routing. Frame TF-IDF as a cheap lexical baseline that makes the latency and accuracy trade-off measurable. |
| 1:00–2:15 | **Dataset Explorer.** Show the 2,000 examples, grouped split and intent distribution. Explain why families stay together, and show a ticket whose secondary intent could mislead the model. |
| 2:15–3:45 | **Fine-Tuning Lab.** Show training and validation loss, validation macro F1, and the selected epoch. State that all pretrained DistilBERT layers and the new classification head receive gradients. |
| 3:45–5:45 | **JEPA Research Lab.** Walk through the hidden span, visible context, positional predictor, stop-gradient teacher and EMA. Show latent loss and feature spread, then explain the frozen probe and PCA view. |
| 5:45–7:15 | **Model Benchmark.** Compare the same held-out set and warmed latency. Show an error and the matched control. Explain that the JEPA objective learned to predict latent states, while this run’s downstream probe underperformed the control. |
| 7:15–8:30 | **Live Ticket Router.** Load “I was charged twice and want my money back.” Compare all three scores and route the selected model to Refund Operations at 0.55. Set the threshold to 1.00 and resubmit to demonstrate human review. The preset appears in training, so enter a fresh paraphrase for a less scripted demonstration. |
| 8:30–9:00 | **Research Conclusions.** Present the observed trade-offs and propose validation on independently labeled real tickets, multiple family splits, and review coverage versus errors. |

### Likely questions

- **Why use macro F1?** It gives each routing intent equal weight. Accuracy can hide a weak class, especially when the grouped test split contains 25 examples for some classes and 50 for others.
- **How do we know the teacher does not learn by backpropagation?** Its parameters have gradients disabled, its states are computed under `no_grad`, and tests verify that its weights change only after EMA.
- **Why did JEPA underperform?** This experiment shows a regression for the chosen objective, pooling, corpus size and hyperparameters. It cannot identify the cause. A masking, pooling and regularization ablation across multiple seeds would test candidate explanations.
- **Does a score of 90% mean the route is correct 90% of the time?** The model assigns that score to its preferred class. The probe is calibrated on synthetic validation data, and reliability on real traffic remains unknown.
- **What would support a production decision?** Evaluate de-identified real tickets with independently assigned primary and secondary intents, measure misroutes and human-review workload, and choose a threshold for the operational cost.
