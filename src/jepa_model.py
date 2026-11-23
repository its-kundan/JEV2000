"""Text JEPA with contiguous span masking and an EMA teacher."""
import copy

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DistilBertModel


def masked_batch(ids, attention, special, mask_id, generator, fraction=.3):
    """Sample positions using sequence structure alone. Sample on CPU for all devices."""
    selections = []
    for valid in (attention.bool() & ~special.bool()):
        eligible = valid.nonzero(as_tuple=True)[0]
        count = min(len(eligible) - 1, max(1, round(len(eligible) * fraction)))
        if count < 1:
            raise ValueError("JEPA requires at least two ordinary tokens per ticket")
        start = torch.randint(len(eligible) - count + 1, (), generator=generator).item()
        selections.append(eligible[start:start + count])
    positions = torch.nn.utils.rnn.pad_sequence(selections, batch_first=True)
    active = positions != 0  # position zero is DistilBERT's leading special token
    context = ids.clone()
    row_indices, target_indices = active.nonzero(as_tuple=True)
    context[row_indices, positions[row_indices, target_indices]] = mask_id
    return context, positions, active


def mean_pool(hidden, mask):
    return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


class TextJEPA(nn.Module):
    def __init__(self, encoder, max_length):
        super().__init__()
        self.context = encoder
        self.target = copy.deepcopy(encoder).requires_grad_(False).eval()
        width = encoder.config.dim
        self.position = nn.Embedding(max_length, width)
        self.predictor = nn.Sequential(nn.LayerNorm(width * 2), nn.Linear(width * 2, width),
                                       nn.GELU(), nn.Linear(width, width))

    def train(self, mode=True):
        super().train(mode)
        self.target.eval()
        return self

    def forward(self, context, original, attention, positions):
        hidden = self.context(input_ids=context, attention_mask=attention).last_hidden_state
        pooled = mean_pool(hidden, attention)
        condition = torch.cat((pooled[:, None].expand(-1, positions.shape[1], -1),
                               self.position(positions)), dim=-1)
        with torch.no_grad():
            targets = self.target(input_ids=original, attention_mask=attention).last_hidden_state
            targets = targets.gather(1, positions.unsqueeze(-1).expand(-1, -1, targets.shape[-1]))
        return self.predictor(condition), targets, pooled

    @torch.no_grad()
    def update_target(self, momentum):
        for teacher, student in zip(self.target.parameters(), self.context.parameters(), strict=True):
            if student.requires_grad:
                teacher.lerp_(student, 1 - momentum)


def objective(prediction, target, active, pooled):
    distance = 1 - F.cosine_similarity(prediction, target, dim=-1)
    latent_loss = (distance * active).sum() / active.sum().clamp_min(1)
    context_std = (pooled.var(dim=0, unbiased=False) + 1e-4).sqrt()
    variance_loss = F.relu(.2 - context_std).mean()
    return latent_loss + .1 * variance_loss, latent_loss, context_std.mean()


def slim_encoder(source, layers, revision):
    encoder = DistilBertModel.from_pretrained(source, revision=revision, attn_implementation="eager")
    encoder.transformer.layer = encoder.transformer.layer[:layers]
    encoder.config.n_layers = layers
    encoder.embeddings.requires_grad_(False)
    return encoder
