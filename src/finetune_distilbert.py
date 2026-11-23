import argparse
import time

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.common import LABELS, MODEL, OUT, REVISION, device_run, fingerprint, read, train_args, validate_args, write_json


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
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=REVISION, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)},
        attn_implementation="eager").to(device)
    train = batches(read("train"), tokenizer, args, shuffle=True)
    val = batches(read("validation"), tokenizer, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = min(len(train), args.max_steps or len(train))
    total = steps * args.epochs
    schedule = get_linear_schedule_with_warmup(optimizer, total // 10, total)
    folder = OUT / "distilbert"
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            schedule.step()
            size = len(batch.labels)
            train_loss += loss.item() * size
            count += size
        model.eval()
        val_loss, predictions, gold = 0., [], []
        with torch.inference_mode():
            for batch in val:
                batch = batch.to(device)
                result = model(**batch)
                val_loss += result.loss.item() * len(batch.labels)
                predictions.extend(result.logits.argmax(dim=-1).tolist())
                gold.extend(batch.labels.tolist())
        score = f1_score(gold, predictions, labels=range(len(LABELS)), average="macro", zero_division=0)
        row = dict(epoch=epoch, train_loss=train_loss / count, validation_loss=val_loss / len(gold),
                   validation_accuracy=accuracy_score(gold, predictions), validation_macro_f1=score,
                   lr=schedule.get_last_lr()[0], epoch_seconds=time.perf_counter() - start_epoch,
                   device=str(device), optimizer_steps=steps)
        history.append(row)
        if score > best_score:
            best_score, best_epoch = score, epoch
            model.save_pretrained(folder, safe_serialization=True)
            tokenizer.save_pretrained(folder)
        write_json(folder / "history.json", history)
        print(row, flush=True)
        if epoch - best_epoch >= args.patience:
            break
    write_json(folder / "run.json", dict(config=vars(args), revision=REVISION, best_epoch=best_epoch,
               seconds=time.perf_counter() - started, device=str(device),
               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
               smoke=args.max_steps is not None, data=fingerprint("train", "validation")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    train_args(parser, 3, 2e-5)
    parser.add_argument("--model", default=MODEL, help="DistilBERT Hub ID or local pretrained snapshot")
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()
    validate_args(parser, args)
    if args.patience < 1:
        parser.error("Patience must be positive")
    device_run(lambda device: fit(args, device), args.device, args.seed)


if __name__ == "__main__":
    main()
