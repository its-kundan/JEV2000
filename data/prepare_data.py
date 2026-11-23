"""Download an optional Bitext overlap set to study label coverage."""
import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote
from urllib.request import urlretrieve

import pandas as pd
from datasets import load_dataset

from src.common import DATA, ROOT, write_json

DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
CSV = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
INTENT_MAP = {
    "check_invoice": "Billing",
    "get_invoice": "Billing",
    "payment_issue": "Billing",
    "get_refund": "Refund",
    "track_refund": "Refund",
    "recover_password": "Account Access",
    "track_order": "Order Tracking",
    "contact_customer_service": "General Inquiry",
    "contact_human_agent": "General Inquiry",
}


def map_tickets(frame):
    matched = frame.loc[frame.intent.isin(INTENT_MAP), ["intent", "instruction"]].copy()
    matched = matched.rename(columns={"instruction": "text", "intent": "source_intent"})
    matched["text"] = matched.text.str.strip()
    matched = matched.dropna(subset=["text"])
    matched = matched[matched.text.ne("")].copy()
    matched["label"] = matched.source_intent.map(INTENT_MAP)
    matched["normalized"] = matched.text.str.lower().str.replace(r"\s+", " ", regex=True)
    conflicts = matched.groupby("normalized").label.transform("nunique") > 1
    return matched.loc[~conflicts].drop_duplicates("normalized").drop(columns="normalized")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Hub revision or immutable commit SHA")
    args = parser.parse_args()
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{quote(args.revision, safe='')}/{CSV}"
    with TemporaryDirectory() as temp:
        path = Path(temp) / CSV
        urlretrieve(url, path)
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        source = load_dataset("csv", data_files=str(path), split="train",
                              cache_dir=str(ROOT / ".cache/datasets"))
        frame = map_tickets(source.to_pandas())
    if frame.empty:
        raise ValueError("No Bitext intent matched the routing label map")
    folder = DATA / "bitext_overlap"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "tickets.csv", index=False)
    counts = frame.label.value_counts().to_dict()
    write_json(folder / "provenance.json", dict(source=DATASET, revision=args.revision,
               url=url, original_sha256=source_hash,
               license="CDLA-Sharing-1.0", upstream_rows=len(source), counts=counts,
               limitation="Generated/hybrid upstream data, partial label coverage, latent paraphrase families unknown. Excluded from the benchmark."))
    print(counts)


if __name__ == "__main__":
    main()
