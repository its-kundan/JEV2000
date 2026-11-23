import argparse
import itertools
import math
import time

import torch
from safetensors.torch import save_model
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.common import MODEL, OUT, REVISION, device_run, fingerprint, read, train_args, validate_args, write_json
from src.jepa_model import TextJEPA, masked_batch, objective, slim_encoder


def momentum_at(step, total):
    return .996 + (.999 - .996) * (1 - math.cos(math.pi * step / max(1, total - 1))) / 2


def fit(args, device):
    started = time.perf_counter()
    folder = OUT / "jepa"
    reference = OUT / "reference"
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=REVISION)
    encoder = slim_encoder(args.model, args.layers, REVISION)
    # Capture the exact frozen representation used as the control before JEPA updates.
    encoder.save_pretrained(reference, safe_serialization=True)
    tokenizer.save_pretrained(reference)
    model = TextJEPA(encoder, args.max_length).to(device)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)

    def tokenize(texts):
        return tokenizer(texts, truncation=True, padding=True, max_length=args.max_length,
                         return_special_tokens_mask=True, return_tensors="pt")

    texts = read("train").text.tolist()
    loader = DataLoader(texts, batch_size=args.batch_size, shuffle=True, collate_fn=tokenize,
                        generator=torch.Generator().manual_seed(args.seed))
    steps = min(len(loader), args.max_steps or len(loader))
    total = steps * args.epochs
    schedule = get_linear_schedule_with_warmup(optimizer, total // 10, total)
    history = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        totals = torch.zeros(5)
        for tokens in itertools.islice(loader, steps):
            original, attention = tokens.input_ids, tokens.attention_mask
            context, positions, active = masked_batch(original, attention, tokens.special_tokens_mask,
                                                     tokenizer.mask_token_id, generator, args.mask_fraction)
            context, original, attention, positions, active = [
                tensor.to(device) for tensor in (context, original, attention, positions, active)]
            optimizer.zero_grad(set_to_none=True)
            predicted, target, pooled = model(context, original, attention, positions)
            loss, latent, spread = objective(predicted, target, active, pooled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            optimizer.step()
            schedule.step()
            momentum = momentum_at(step, total)
            model.update_target(momentum)
            step += 1
            with torch.no_grad():
                target_vectors = target[active]
                target_std = target_vectors.var(dim=0, unbiased=False).sqrt().mean()
                pred_std = predicted[active].var(dim=0, unbiased=False).sqrt().mean()
                totals += torch.stack((loss, latent, spread, target_std, pred_std)).detach().cpu()
        row = dict(zip(("loss", "latent_loss", "context_std", "target_std", "prediction_std"),
                       (totals / steps).tolist(), strict=True))
        row.update(epoch=epoch, ema_start=momentum_at(step - steps, total), ema_end=momentum,
                   lr=schedule.get_last_lr()[0], epoch_seconds=time.perf_counter() - epoch_start,
                   device=str(device), optimizer_steps=steps)
        history.append(row)
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "history.json", history)
        print(row, flush=True)
    model.context.save_pretrained(folder / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(folder / "encoder")
    folder.joinpath("research").mkdir(exist_ok=True)
    save_model(model.cpu(), folder / "research/jepa.safetensors")
    write_json(folder / "pretrain.json", dict(config=vars(args), revision=REVISION, device=str(device),
               seconds=time.perf_counter() - started, trainable_parameters=sum(p.numel() for p in params),
               smoke=args.max_steps is not None, data=fingerprint("train")))


def main():
    parser = argparse.ArgumentParser()
    train_args(parser, 3, 5e-5)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--layers", type=int, choices=range(1, 7), default=2)
    parser.add_argument("--mask-fraction", type=float, default=.3)
    args = parser.parse_args()
    validate_args(parser, args)
    if not 0 < args.mask_fraction < 1:
        parser.error("Mask fraction must lie between zero and one")
    device_run(lambda device: fit(args, device), args.device, args.seed)


if __name__ == "__main__":
    main()
