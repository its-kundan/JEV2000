import pandas as pd

from src.common import ROOT

DATA = ROOT / "data/generated/support_router"
SPLITS = ("train", "validation", "test")


def read_tickets(split):
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    # North America's region code is also pandas' default missing-value marker.
    return pd.read_csv(DATA / f"{split}.csv", keep_default_na=False)


def main():
    from data.generate_tickets import main as generate

    generate()


if __name__ == "__main__":
    main()
