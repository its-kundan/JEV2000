# System One Intelligent Support Router — presenter script

**Format:** 10-minute manager demo, followed by Q&A. Copy each **On screen** block into a slide, or present directly from this file. Read the **Say** blocks aloud and use **Show** for your clicks. The eight slide sections follow the dashboard navigation.

## Before presenting

From the project folder, start the app using the existing virtual environment:

```bash
source .venv/bin/activate
python3 -m streamlit run app.py --server.address 127.0.0.1
```

Open the URL printed in the terminal. Verify that **DistilBERT** appears under **Department classifier**, that **Model Evaluation** shows three models, and that **Model Fine-Tuning Lab** shows the recorded training curves. Run the two tickets on slide 4 once on the presentation machine, then return to **Executive Overview**. Use the saved checkpoint on stage. The recorded training run took about six minutes.

If the data or checkpoints are missing, prepare them before the demo using the [README](README.md#run-on-macos). Keep the [saved benchmark](artifacts/support_router/metrics/benchmark.json) or screenshots of your rehearsal as a fallback. The installed app with its generated data and checkpoints routes tickets locally without a key. The initial dependency installation and pretrained-weight download may require internet access.

## Slide 1 · What the router does · 0:00–1:00

**On screen**

- One support ticket, 2,000 possible queues
- Ticket → Department → Shortlist → Typed decision → Queue or human review
- Local inference and evaluation

**Show** — Open **Executive Overview** and point to the pipeline from left to right.

**Say**

> Imagine a customer says they were charged twice. We need to get that ticket to the right team for their product and region, or send it to a person if we lack enough evidence. This prototype starts with 2,000 possible queues. It predicts the department, retrieves candidate queues, and produces a structured decision that a routing policy can check.
>
> TypeSafe’s System One approach inspired the typed decision layer. In this demo, our classifier and deterministic rules run on this machine. The system returns a queue ID, urgency, severity, and a review flag. There’s no account, API key, or usage charge.

**Transition**

> First, here’s where those 2,000 choices come from.

## Slide 2 · How we built the catalog and dataset · 1:00–2:00

**On screen**

- 8 departments × 5 issue categories × 10 products × 5 regions = 2,000 queues
- 12,000 synthetic tickets, split 8,000 train / 2,000 validation / 2,000 test
- Entire wording families held out

**Show** — Open **Dataset & Destinations**. Point to the hierarchy, then paste `BILLING-DUPLICATE-CHARGE-APAC-001` into **Search by ID, issue, region, or product**. Scroll to the split counts.

**Say**

> Every queue has a stable identifier and catalog metadata. This one is Billing, duplicate charge, APAC, for Atlas Cloud. The dataset gives us reproducible examples where the correct queue is known.
>
> We grouped related phrasings before splitting, so validation and test each get a wording family that the models didn’t train on. The same products and regions appear across splits by design. This is a test of the prototype on synthetic tickets. We still need representative customer tickets to measure real-world performance.

## Slide 3 · What we trained · 2:00–3:15

**On screen**

- Baseline · TF-IDF and logistic regression
- Control · pretrained DistilBERT encoder with an initialized classification head
- Fine-tuned model · pretrained encoder and head trained on eight departments

**Show** — Open **Model Fine-Tuning Lab**. Show the recorded validation macro F1 curve and **Best validation epoch**. Expand **Completed run configuration** if someone asks about the device or batch size. Keep the saved checkpoint loaded for the demo.

**Say**

> Our baseline learns from words and two-word phrases. DistilBERT starts with a pretrained encoder that can use words in context. We attach a new eight-class head and train on labeled support tickets. The initialized-head control shows how that same pretrained encoder performs before supervised training on our department labels.
>
> We ran three epochs on a Mac CPU with batch size 32. Epoch one had the best validation macro F1 at about 86.8 percent, so that’s the checkpoint we kept. We used validation to select the checkpoint and fit confidence calibration, then scored the chosen model on test tickets.

## Slide 4 · Live routing and human review · 3:15–5:20

**On screen**

- 2,000 catalog queues → 250 in the predicted department → 15 candidates
- Clear ticket → Billing queue
- Ambiguous ticket or higher threshold → Human review

**Show** — Open **Live Ticket Router**. Choose **DistilBERT**, set **Top-K destinations** to **15**, and set **Confidence threshold** to **0.70**. Submit these tickets in order using **Route ticket** after each change.

| Run | Paste into **Customer ticket** | **Customer region** | Expected with the saved checkpoint |
| --- | --- | --- | --- |
| A | I was charged twice and I need an urgent refund. | Unspecified | Refund, approximately 98% department confidence. Urgent, severity 4/5. Human review. |
| B | Atlas Cloud APAC: charged twice for one order. | APAC | Billing, approximately 91% department confidence. Auto-route to `BILLING-DUPLICATE-CHARGE-APAC-001`. |
| Optional C | Reuse ticket B and raise the threshold to 0.95 | APAC | The proposed queue stays the same. The final action becomes human review because department confidence is below the threshold. |

**Say after A**

> The model is confident that Refund is the main department. The simulator still flags the ticket for review. We have a duplicate-charge complaint and a refund request, and the ticket gives us no product or region. Department confidence only tells us about the classifier’s department prediction. Queue selection has its own evidence and review rules.

**Say after B**

> This ticket gives us the issue, product, and region. The Billing prediction narrows the catalog to 250 queues. Retrieval ranks those queues by text similarity and product and region matches. We keep the top 15, and the policy accepts the duplicate-charge queue for Atlas Cloud in APAC.

**Say if showing C**

> The classifier and proposed queue stayed the same. I raised the threshold, so the policy now sends this ticket for review. That gives the operator a way to control how much the system automates.

**Transition** — Submit B again at **0.70** if you showed C. Click **Inspect the decision**.

## Slide 5 · What a typed decision contains · 5:20–6:15

**On screen**

- State · ticket, predicted department, confidence, shortlist
- Decision · department, urgency, severity, escalation, destination
- Validation · answer types and shortlist membership

**Show** — In **Decision Inspector**, point to **Structured state**, **Typed questions**, and **Validated local decision**. Open **Department probabilities**.

**Say**

> The destination must be one of the candidate IDs. Urgency and escalation are booleans, and severity is a number from one through five. We validate the decision before accepting a route. The department probabilities come from the classifier. The local rules use the shortlist to choose a destination and can flag a ticket for human review. The proposed destination and the final routing action are both visible here.

## Slide 6 · What the classifier improved · 6:15–7:40

**On screen**

| Model | Test department macro F1 |
| --- | ---: |
| TF-IDF baseline | 64.26% |
| DistilBERT with initialized head | 4.81% |
| Fine-tuned DistilBERT | 88.07% |

**Show** — Open **Model Evaluation**, choose **test**, and select **DistilBERT** in **Inspect model**. Point to the confusion matrix and reliability diagram.

**Say**

> These results are on the same 2,000 held-out synthetic tickets. Fine-tuning improved department macro F1 by about 24 percentage points over the baseline. Macro F1 weights every department equally.
>
> The confusion matrix shows which actual departments the model mixes up. The reliability diagram compares the model’s stated confidence with the fraction it gets right. That helps us interpret the threshold. I’ll use the next page to show how often that better department model leads to the right final queue.

## Slide 7 · What the router gets right and wrong · 7:40–9:10

**On screen**

Fine-tuned DistilBERT plus the local simulator, K=15, threshold 0.70, on 2,000 synthetic test tickets:

- **88.05%** had the correct queue in the shortlist
- **64.00%** were auto-routed, including **44.45%** correctly and **19.55%** incorrectly
- **36.00%** went to human review
- **69.45%** of auto-routed tickets went to the right queue

**Show** — Open **Routing Experiments**. Point to **Candidate recall by shortlist size** and the DistilBERT row at K=15. If time allows, use **Run a routing policy**, keep the seeded sample at **25**, check **Compare thresholds 0.60, 0.70, 0.80, 0.90, 0.95**, and click **Run experiment**. Use the full test results for your headline numbers.

**Say**

> The correct queue was in the shortlist for 88 percent of tickets, but the ranking rules still picked the wrong issue in some of those shortlists. At our current threshold we automatically routed 64 percent of all tickets. About 44 percent of all tickets went to the right queue automatically, 20 percent went to the wrong one, and 36 percent went to review.
>
> Increasing K didn’t improve recall in this synthetic run. Every test ticket names a product and region, and the correct queue was already in the top five when the department was correct. Increasing the confidence threshold is a different control. It can send more tickets to review, but it can’t recover a queue excluded by the department filter.

## Slide 8 · Recommendation and close · 9:10–10:00

**On screen**

- Measure routing on representative, labeled tickets
- Compare single-department and multi-department retrieval
- Improve issue ranking and choose a threshold based on misroutes and review workload

**Show** — Open **Research Findings**. End on **Limitations and future work**.

**Say**

> This prototype demonstrates local routing, typed decisions, and a working human-review policy. On the synthetic test, the model classified departments better after fine-tuning, but about one in five tickets still went to the wrong queue automatically at the demo threshold. That’s too high to justify unattended routing on this evidence.
>
> My recommendation is a human-reviewed pilot with a representative labeled sample. I’d compare retrieval across more than one likely department, improve issue ranking, and measure how many misroutes we can avoid for a given review workload. That’s the next decision I’m asking for today.
>
> Thank you. I’m happy to take questions.

## Q&A · answers to say aloud

| If someone asks | Say |
| --- | --- |
| **What problem does this solve?** | “It proposes a queue from a large catalog and sends uncertain cases to a person. That gives support staff a starting point and exposes why the system chose it.” |
| **What is System One? What is Jev?** | “TypeSafe uses System One to describe models that return structured decisions for software. Jev is its hosted model. Our local simulator demonstrates the idea with explicit rules.” |
| **Is Jev the same as JEPA? Did you fine-tune Jev?** | “JEPA is a representation-learning architecture. Jev is TypeSafe’s hosted decision model. We fine-tuned DistilBERT for department classification.” |
| **Does this need an API key or have a usage cost?** | “Routing and evaluation run locally. There’s no API account or per-request charge. A fresh setup may need internet to install packages and download the pretrained weights.” |
| **Why use two stages instead of predicting all 2,000 queues directly?** | “There are only four training examples per exact queue. Department labels have more data. Then catalog metadata handles the individual queues. The cost is that a wrong department can exclude the correct queue.” |
| **What does fine-tuning do?** | “We start with a pretrained language encoder and update it and a new head using our labeled support tickets. The final head predicts eight departments.” |
| **Why keep the baseline and initialized-head control?** | “The baseline tells us what a cheap bag-of-words model can do. The initialized head tells us how the pretrained encoder and untrained head behave before learning the routing labels.” |
| **How does retrieval work?** | “We filter active queues by the predicted department, rank descriptions and keywords by TF-IDF cosine similarity, and boost product and region matches. Then we keep K candidates.” |
| **Can the system invent a queue ID?** | “The validated decision can select only a queue ID from the actual shortlist. That still leaves room for picking the wrong valid queue, which we measure.” |
| **What does 91% confidence mean?** | “It’s the classifier’s calibrated probability for Billing on that ticket. It does not estimate the probability that our exact queue choice is correct.” |
| **What is calibration? How do you check it?** | “We fit one temperature using validation predictions. On test, we compare confidence with observed correctness in groups. ECE summarizes those gaps. Calibration can change when the incoming ticket mix changes.” |
| **What if the correct queue isn’t in the shortlist?** | “This policy can’t select it. That is why we measure recall separately and why I’d test retrieval across multiple likely departments.” |
| **Will a 95% threshold solve the misrouting problem?** | “It can reduce automation or send more tickets to review. The model can still be confidently wrong, and the queue ranking can still be wrong when the department is right.” |
| **Why did coverage go up and misrouting also go up compared with the baseline?** | “At 0.70, the fine-tuned model auto-routed 64% versus 43% for the baseline. It routed a slightly larger fraction of those automatic decisions correctly, but it also sent more total tickets through the policy, so the share of all tickets misrouted rose from 13.70% to 19.55%.” |
| **Why is the routing accuracy lower than department accuracy?** | “A department can be right while the exact product, region, or issue queue is wrong. Also, end-to-end correct auto-routing counts review cases in the denominator.” |
| **How fast is it?** | “The saved benchmark shows warmed single-ticket model and end-to-end timings separately. On our Mac the department baseline is sub-millisecond and the fine-tuned classifier is around six milliseconds. The figures depend on hardware, checkpoint, and timing method.” |
| **Could the test score be inflated by synthetic data?** | “Yes. We hold out entire wording families, but catalog terms recur and the generated tickets have a narrow range of styles. I’d measure on a representative human-labeled sample before drawing conclusions about customers.” |
| **How do new queues get added?** | “A new queue within an existing department can be added through catalog metadata and a refreshed retrieval index. A new department changes the classifier’s label space and requires retraining and evaluation.” |
| **What changes if you put this into production?** | “I’d begin with a human-reviewed pilot. We’d need access controls for tickets, audit logs, a queue integration, and monitoring of errors and review volume. This demo proposes a route on screen.” |

## If the live demo gets stuck

- **The live result differs from rehearsal.** Check the classifier, region, K, and threshold. Say, “Let’s look at the evidence and why the policy made that choice.” Open the inspector and show the actual result.
- **DistilBERT is missing.** Use **Baseline** for the mechanics of the demo. Say, “This instance is using the local baseline. The comparative figures come from our saved evaluation.” Skip the promised queue for ticket B.
- **The dashboard stops responding.** Show your saved screenshot or read the benchmark slide. Say, “Here’s the recorded run on the held-out set. I’ll use it to show where the system succeeds and where it needs review.”
- **You’re running out of time.** Show ticket B, then ticket A. Go directly to the routing results and finish with the human-reviewed pilot recommendation.
