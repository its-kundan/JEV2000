import argparse
from random import Random

import pandas as pd

from data.generate_destinations import TAXONOMY, build_destinations
from data.prepare_dataset import DATA, SPLITS
from src.common import write_json
from src.router_config import CONFIG

TEMPLATES = (
    "{issue}. I'm using {product} in {region}. {extra}",
    "Hello support, our {region} team has a question on {product}. Here's what's happening: {issue}. {extra}",
    "{product} / {region}: {issue}. {extra}",
    "hey, {issue} on {product} ({region}) {extra}",
    "Could a specialist check {product} for our office in {region}? The reason I'm writing: {issue}. {extra}",
    "From {region}, about {product} — {issue}. {extra}",
)


def build_tickets(count=12000, seed=42):
    if not 240 <= count <= 12000:
        raise ValueError("Ticket count must be between 240 and 12,000")
    random = Random(seed)
    families = list(range(len(TEMPLATES)))
    random.shuffle(families)
    assignments = dict(zip(families, ["train"] * 4 + ["validation", "test"]))
    catalog = build_destinations()
    buckets = {}
    for row in catalog:
        for family in families:
            buckets.setdefault((row["department"], row["subcategory"], family), []).append(row)
    for bucket in buckets.values():
        random.shuffle(bucket)
    keys = list(buckets)
    random.shuffle(keys)
    rows = []
    for index in range(count):
        key = keys[index % len(keys)]
        department, issue, family = key
        destination = buckets[key].pop()
        urgent = random.random() < .2
        multi = random.random() < .13
        extra = "Urgent, this is blocking our team." if urgent else ""
        if multi:
            secondary = {"Billing": "I also have a question about logging in.",
                         "Account Access": "I also need to check my invoice."}
            extra += " " + secondary.get(department, "I also need to update my credit card.")
        text = TEMPLATES[family].format(issue=TAXONOMY[department][issue][family],
                                      product=destination["product"], region=destination["region"],
                                      extra=extra.strip()).strip()
        rows.append(dict(ticket_id=f"SYN-{index+1:05}", ticket_text=text, department=department,
                         subcategory=issue, destination_id=destination["destination_id"],
                         urgency=urgent, requires_human_review=multi or issue == "Unclear request",
                         region=destination["region"], template_family=f"narrative-{family}",
                         split=assignments[family], synthetic=True))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate grouped synthetic support tickets.")
    parser.add_argument("--count", type=int, default=CONFIG["ticket_count"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    args = parser.parse_args()
    if not 240 <= args.count <= 12000:
        parser.error("--count must be between 240 and 12,000")
    data = build_tickets(args.count, args.seed)
    DATA.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        data[data.split == split].to_csv(DATA / f"{split}.csv", index=False)
    summary = dict(seed=args.seed, synthetic=True, count=len(data), partitions=data.split.value_counts().to_dict())
    write_json(DATA / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
