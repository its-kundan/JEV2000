import argparse
import fcntl
import os
import time
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data.generate_destinations import TAXONOMY
from data.prepare_dataset import read_tickets
from src.common import MODEL, REVISION, device_run, train_args, validate_args, write_json
from src.confidence_calibration import fit_temperature
from src.router_config import DISTILBERT, INITIAL, LOGS, TRAIN_LOCK

LABELS = list(TAXONOMY)


def read(split):
    return read_tickets(split).rename(columns={"ticket_text": "text", "department": "label"})



def batches(frame, tokenizer, args, shuffle=False):
    def collate(rows):
        texts, labels = zip(*rows)
        batch = tokenizer(list(texts), padding=True, truncation=True, max_length=args.max_length,
                          return_tensors="pt")
        batch["labels"] = torch.tensor(labels)
        return batch

    return DataLoader(list(zip(frame.text, frame.label.map(LABELS.index))), batch_size=args.batch_size,
                      shuffle=shuffle, generator=torch.Generator().manual_seed(args.seed), collate_fn=collate)


def fit(args, device):
    suffix = "_smoke" if args.max_steps else ""
    destination = DISTILBERT.with_name(DISTILBERT.name + suffix)
    initial = INITIAL.with_name(INITIAL.name + suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp)
        train_and_save(args, device, stage / destination.name, stage / initial.name)
        # Keep the previous completed run available until its replacement is ready.
        backups = {}
        published = []
        try:
            for path in (destination, initial):
                if path.exists():
                    backups[path] = path.rename(stage / f"{path.name}.previous")
                (stage / path.name).rename(path)
                published.append(path)
        except OSError:
            for path in published:
                path.rename(stage / path.name)
            for path, backup in backups.items():
                backup.rename(path)
            raise


def train_and_save(args, device, folder, initial_folder):
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=REVISION, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)},
        attn_implementation="eager").to(device)
    if not 0 <= args.freeze_layers <= len(model.distilbert.transformer.layer):
        raise ValueError("Freeze layers must fit the DistilBERT encoder")
    if args.freeze_layers:
        model.distilbert.embeddings.requires_grad_(False)
        for layer in model.distilbert.transformer.layer[:args.freeze_layers]:
            layer.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    model.save_pretrained(initial_folder, safe_serialization=True)
    tokenizer.save_pretrained(initial_folder)
    write_json(initial_folder / "run.json", dict(config=vars(args), temperature=1, seconds=0,
                                                trainable_parameters=0, device=str(device)))
    train = batches(read("train"), tokenizer, args, shuffle=True)
    val = batches(read("validation"), tokenizer, args)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps = min(len(train), args.max_steps or len(train))
    total = steps * args.epochs
    schedule = get_linear_schedule_with_warmup(optimizer, total // 10, total)
    run_log = LOGS / f"{folder.name}_history.json"
    history, best_score, best_epoch = [], -1, 0
    for epoch in range(1, args.epochs + 1):
        start_epoch = time.perf_counter()
        model.train()
        train_loss, count = 0., 0
        for index, batch in enumerate(train):
            if index >= steps:
                break
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            schedule.step()
            size = len(batch.labels)
            train_loss += loss.item() * size
            count += size
        model.eval()
        val_loss, predictions, gold, logits = 0., [], [], []
        with torch.inference_mode():
            for batch in val:
                batch = batch.to(device)
                result = model(**batch)
                val_loss += result.loss.item() * len(batch.labels)
                predictions.extend(result.logits.argmax(dim=-1).tolist())
                gold.extend(batch.labels.tolist())
                logits.extend(result.logits.cpu().tolist())
        score = f1_score(gold, predictions, labels=range(len(LABELS)), average="macro", zero_division=0)
        row = dict(epoch=epoch, train_loss=train_loss / count, validation_loss=val_loss / len(gold),
                   validation_accuracy=accuracy_score(gold, predictions), validation_macro_f1=score,
                   lr=schedule.get_last_lr()[0], epoch_seconds=time.perf_counter() - start_epoch,
                   device=str(device), optimizer_steps=steps, trainable_parameters=trainable_count)
        history.append(row)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_logits, best_labels = np.array(logits), np.array(gold)
            model.save_pretrained(folder, safe_serialization=True)
            tokenizer.save_pretrained(folder)
        write_json(run_log, history)
        print(row, flush=True)
        if epoch - best_epoch >= args.patience:
            break
    write_json(folder / "run.json", dict(config=vars(args), revision=REVISION, best_epoch=best_epoch,
               seconds=time.perf_counter() - started, device=str(device),
               trainable_parameters=trainable_count, smoke=args.max_steps is not None,
               temperature=fit_temperature(best_logits, best_labels), history=history))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    train_args(parser, 3, 2e-5)
    parser.add_argument("--model", default=MODEL, help="DistilBERT Hub ID or local pretrained snapshot")
    parser.add_argument("--freeze-layers", type=int, default=0,
                        help="Freeze embeddings and the first N transformer blocks when N > 0.")
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()
    validate_args(parser, args)
    if args.patience < 1:
        parser.error("Patience must be positive")
    LOGS.mkdir(parents=True, exist_ok=True)
    with TRAIN_LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, "A training job is already running.\n")
        threads = int(os.environ.get("ROUTER_TORCH_THREADS", "4"))
        torch.set_num_threads(threads)
        device_run(lambda device: fit(args, device), args.device, args.seed)


if __name__ == "__main__":
    main()
