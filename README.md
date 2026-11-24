# System One Intelligent Support Router

**2,000 Destination Decision Engine** · Local Streamlit research POC

A support ticket passes through an eight-department classifier, queue retrieval, five typed questions, and a policy that chooses a destination or sends the ticket for human review. The entire routing and evaluation flow runs locally with a deterministic decision simulator. It requires no account, API key, or paid service.

## Run on macOS

Use Python 3.12 and run commands from this repository. The dashboard and the local baseline work without network access after dependency installation. DistilBERT needs its pretrained weights on the first training run.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 -m data.generate_destinations
python3 -m data.generate_tickets
python3 -m src.baseline_model
python3 -m src.train_distilbert
python3 -m src.benchmarking
python3 -m pytest
python3 -m streamlit run app.py --server.address 127.0.0.1
```

Open [Streamlit on port 8501](http://127.0.0.1:8501). For a baseline-only demo, skip neural training and run `python3 -m src.benchmarking --baseline-only`. The test suite trains and reloads a tiny checkpoint created from a local fixture for its neural smoke test. It needs neither a model download nor an account.

### Training options and artifacts

```sh
# Train from a locally cached pretrained snapshot
python3 -m src.train_distilbert --model .cache/distilbert --device cpu --batch-size 32

# Freeze embeddings and the first five of six transformer blocks
python3 -m src.train_distilbert --freeze-layers 5

# One optimizer step with an available pretrained snapshot
python3 -m src.train_distilbert --model .cache/distilbert --epochs 1 --max-steps 1
```

The full run defaults to three epochs, AdamW at 2e-5, weight decay 0.01, sequence length 128, warmup and linear decay, gradient clipping at 1, a seeded sampler, and early stopping after two epochs without validation macro F1 improvement. The best validation checkpoint is restored via its saved weights. With zero frozen blocks, the embeddings and all six transformer blocks train along with the new classification layers. Freezing six blocks leaves the classification layers trainable.

Auto device selection checks CUDA, MPS, then CPU. CPU can take several minutes or longer depending on hardware and batch size. MPS and CUDA can accelerate training when supported. The recorded run used macOS CPU. MPS and CUDA were unavailable in that run. The dashboard training lab launches the same CLI with a process lock and captures the CLI log.

- `data/generated/support_router/` contains the generated catalog and split CSVs. `config.yaml` sets the seed, ticket count, default shortlist size, and default confidence threshold.
- `artifacts/support_router/models/` contains the trained baseline, initialized-head control, and best fine-tuned checkpoint. Step-capped checkpoints use a `_smoke` suffix.
- `artifacts/support_router/logs/` holds per-epoch loss, accuracy, macro F1, learning rate, time, parameter count, and device.
- `artifacts/support_router/metrics/` holds synthetic evaluation results and data fingerprints. Start with `benchmark.json`. Generated data and artifacts are Git-ignored.

## Design and evaluation

The catalog has eight departments with five issue categories each. Each issue has a distinct queue for every combination of ten synthetic products and five regions. Each catalog entry includes a stable destination ID, team, description, supported issues, region, priority support, routing keywords, and active status.

The generator creates 12,000 synthetic tickets by default. Set `--count` between 240 and 12,000 to change the size. Six narrative templates each have dedicated issue paraphrases. Four entire narrative families go into training and one each into validation and test. The default produces **8,000 / 2,000 / 2,000** tickets. Both validation and test use unseen narrative syntax and phrasings. Product and region terms recur intentionally. Queue labels are combinations of issue, product and region. In the full synthetic set, each queue has four training examples, which is too sparse to draw conclusions about flat 2,000-way learning. Some tickets include secondary intents, ambiguous requests, misspellings, or urgency.

1. **Classify.** The word unigram/bigram TF-IDF pipeline trains logistic regression on departments. DistilBERT trains on the same labels. The experiment includes the same pretrained encoder with its newly initialized head before any supervised updates.
2. **Retrieve.** Filter active destinations by the predicted department and compute cosine similarity between the ticket and catalog documents. Catalog documents contain department, issue, description and support-team keywords defined per queue. The narrative paraphrases stay out of the catalog index. Add deterministic product and region match weights to rank the shortlist. Evaluation uses only ticket text and region as input. Gold IDs are used to score recall afterward.
3. **Decide.** The typed layer outputs department, urgency, severity, escalation, and a destination ID from the shortlist. The simulator chooses the highest-ranked queue and applies text and retrieval rules for urgency, severity and review. Ties, insufficient overlap and some ambiguity cues trigger review. The simulator exposes no destination probability.
4. **Apply policy.** Review is required if department confidence falls below the threshold, the decision disagrees with the department filter, the decision flags review, or validation fails. Output types and shortlist membership are checked before routing.

Validation fits scalar temperatures for baseline and fine-tuned logits. The initialized-head control retains temperature 1. The dashboard shows expected calibration error (ECE), multiclass Brier score (sum across classes), reliability curves, per-class scores and errors. A single validation family makes the calibration data sensitive to wording. Compare validation and test calibration to see how this transfers.

The CLI evaluates common validation and test examples across models. Candidate recall@K is the share of test tickets for which the gold queue appears in the shortlist. Proposed accuracy ignores abstention. Conditional decision accuracy divides correct proposals by the number of tickets with the correct queue retrieved. End-to-end accuracy counts correct automatic routes over all tickets. Misrouting is the share of all tickets automatically sent to the wrong queue. Auto-route accuracy uses only automatically routed tickets as its denominator. The CLI measures warmed single-ticket classifier latency and a separate end-to-end latency sample. The K sweep reuses classifier scores and reports its narrower latency scope. The dashboard supports a threshold sweep with actual end-to-end timing on the same seeded sample.

### Recorded CPU benchmark

The test split contained 2,000 synthetic tickets. DistilBERT trained for three epochs on arm64 macOS CPU using a batch size of 32 and all 66,959,624 trainable parameters. Epoch 1 had the best validation macro F1 (86.76%). The run took 373 seconds on this machine.

| Department model | Test accuracy | Test macro F1 | Median inference | Test ECE before → after |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 64.75% | 64.26% | 0.16 ms | 0.2134 → 0.1376 |
| Pretrained DistilBERT + initialized head | 14.00% | 4.81% | 6.01 ms | 0.0019 → 0.0019 |
| Fine-tuned DistilBERT | 88.05% | 88.07% | 6.20 ms | 0.0388 → 0.0388 |

| Simulator routing at K=15, threshold 0.70 | Recall@15 | Correct proposal given retrieval | Auto-route coverage | Auto-route accuracy | Overall correct auto-route | Misrouted share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 64.75% | 63.47% | 43.00% | 68.14% | 29.30% | 13.70% |
| Fine-tuned DistilBERT | 88.05% | 65.59% | 64.00% | 69.45% | 44.45% | 19.55% |

Recall stayed flat across K=5–50 in this synthetic run. Every test ticket names its product and region, which places its queue within the top five whenever the department is correct. The simulator still picked the wrong issue for some of those tickets. At the same threshold, DistilBERT sent more tickets automatically and misrouted a larger share of all tickets. Median end-to-end latency was 0.42 ms for the baseline and 6.14 ms for DistilBERT over 24 warmed local calls per model. Both sweeps recorded zero schema failures. The initialized head's low ECE reflects its near-uniform probabilities alongside poor accuracy. It offers little routing value.

## Local typed decisions

The **Local Jev-style Simulator** demonstrates structured decisions using explicit routing rules. The department and destination use closed sets of choices, urgency and review are booleans, and severity is an ordinal score from 1 to 5. Confidence comes from the local department classifier. The inspector shows the shortlist, typed questions, actual decision, and review reasons.

TypeSafe's Jev is the hosted System One model that inspired this design. TypeSafe describes it as answering predefined questions with typed probabilities. JEPA refers to Joint-Embedding Predictive Architecture, a separate representation-learning concept. [TypeSafe announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

## Manager demo · 10 minutes

| Time | Page | What to show |
| --- | --- | --- |
| 0:00–1:15 | Executive Overview | Explain TypeSafe's System One concept. Follow the flow from ticket text to a typed routing action and show where the department classifier and decision backend fit. |
| 1:15–2:15 | Dataset & Destinations | Open the hierarchy, search for `BILLING-DUPLICATE-CHARGE-APAC-001`, then show the family-held-out splits. Explain what the synthetic queue ID encodes. |
| 2:15–3:45 | Model Fine-Tuning Lab | Show the three epochs, validation macro F1, selected checkpoint, trainable count and loss curves. Explain that the encoder starts pretrained and that the eight-class head starts random. |
| 3:45–5:15 | Live Ticket Router | Start with “I was charged twice and I need an urgent refund.” Show how missing product or conflicting intent can leave the ticket in review. Then try “Atlas Cloud APAC: charged twice for one order.” Use APAC and a 0.70 threshold. Show the 2,000 → 250 → 15 funnel and the proposed queue. |
| 5:15–6:30 | Decision Inspector | Show the structured input, typed questions, schema status, and validated decision. Explain how department probabilities and review rules affect the final action. |
| 6:30–8:00 | Model Evaluation | Compare TF-IDF, the initialized-head control and fine-tuned DistilBERT. Inspect a confusion matrix and calibration plot on the same test set. |
| 8:00–9:15 | Routing Experiments | Move K between 5 and 50, then run a confidence sweep. Discuss recall, conditional selection, coverage, errors and latency. Raising K can expose the correct queue without changing the simulator's highest-ranked proposal. |
| 9:15–10:00 | Research Findings | Explain the synthetic-data limitations, the department bottleneck, and the next experiment with human-labeled tickets and broader retrieval. |

### Likely technical questions

| Question | Answer |
| --- | --- |
| Why a hierarchy instead of a 2,000-class head? | The synthetic default has four training tickets per queue. Department supervision is denser, and queues can be added through metadata and retrieval. The trade-off is that a wrong department can exclude the right queue. |
| Why DistilBERT, and what is fine-tuning? | It starts with a compact pretrained contextual encoder. Labeled support tickets update the new classification head and the unfrozen encoder using supervised cross-entropy. |
| What exactly is Jev? | TypeSafe's hosted System One model gives typed probabilistic answers to predefined questions about supplied state. |
| How does Jev relate to JEPA? | Jev is a vendor-hosted decision model. JEPA is a term for a representation-prediction architecture and learning objective. |
| Can we fine-tune Jev? | Jev is a hosted service. The supervised training controls in this project apply to DistilBERT. |
| How can we handle 2,000 choices? | Predict a department and retrieve a shortlist. Restrict the destination choice to the 5–50 retrieved candidates. |
| Why measure retrieval separately? | If the gold queue is absent, no valid response can choose it. Conditional selection accuracy records errors that happen after the correct queue survives. |
| What does calibration mean? | Among similar predictions tagged with 80% confidence, roughly 80% should be correct. Validation fits the temperature. The reliability plot and ECE measure transfer to the test split. |
| What happens when confidence is high and the model is wrong? | The policy can still misroute a ticket. The dashboard reports those errors. Manual audits and real-data evaluation are needed to quantify that risk. |
| What does the simulator demonstrate? | It demonstrates structured state, typed questions, shortlist constraints, and review rules. The local classifier supplies department probabilities. |
