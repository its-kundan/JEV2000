import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/generated"
OUT = ROOT / "artifacts"
LABELS = ["Billing", "Refund", "Technical Support", "Account Access",
          "Order Tracking", "Subscription", "General Inquiry", "Product Information"]
MODEL = "distilbert/distilbert-base-uncased"
REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
SEED = 42


def read(split):
    if split not in {"train", "validation", "test"}:
        raise ValueError(split)
    return pd.read_csv(DATA / f"{split}.csv")


def fingerprint(*splits):
    return {s: hashlib.sha256((DATA / f"{s}.csv").read_bytes()).hexdigest() for s in splits}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def load_json(path):
    return json.loads(path.read_text())


def setup_torch(requested, seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    if requested == "auto":
        requested = next(key for key, present in available.items() if present)
    if not available[requested]:
        print(f"{requested} unavailable. Using CPU.", flush=True)
        requested = "cpu"
    return torch.device(requested)


def device_run(function, requested, seed):
    device = setup_torch(requested, seed)
    try:
        return function(device)
    except (NotImplementedError, RuntimeError) as error:
        detail = str(error).lower()
        if device.type == "cpu" or not any(term in detail for term in
                ("not implemented", "not supported", "unsupported", "out of memory")):
            raise
        print(f"{device} operation failed ({error}). Retrying on CPU.", flush=True)
    return function(setup_torch("cpu", seed))


def train_args(parser, epochs, lr):
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-steps", type=int, help="Cap optimizer steps per epoch for a smoke run")


def validate_args(parser, args):
    if min(args.epochs, args.batch_size, args.max_length - 3) < 1 or args.max_length > 512:
        parser.error("Use positive epochs and batch size, and a maximum length between 4 and 512")
    if args.lr <= 0 or args.weight_decay < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Use positive learning rate and step cap, and nonnegative weight decay")
