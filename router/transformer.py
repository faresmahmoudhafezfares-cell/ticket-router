"""Shared helpers for the Transformer models (technique areas C and D)."""
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

MAX_LEN = 48


def batches(texts, labels, tokenizer, batch_size, shuffle, seed=0, extra=None):
    """Yield tokenized mini-batches. `extra` is an optional per-example tensor (teacher logits)."""
    idx = np.arange(len(texts))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, len(idx), batch_size):
        b = idx[start:start + batch_size]
        enc = tokenizer([texts[i] for i in b], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
        out = {"enc": enc}
        if labels is not None:
            out["labels"] = torch.tensor([labels[i] for i in b])
        if extra is not None:
            out["extra"] = extra[torch.as_tensor(b)]
        yield out


@torch.no_grad()
def predict_logits(model, tokenizer, texts, batch_size=128):
    model.eval()
    outs = []
    for b in batches(texts, None, tokenizer, batch_size, shuffle=False):
        outs.append(model(**b["enc"]).logits.float())
    return torch.cat(outs)


def linear_schedule(optimizer, total_steps, warmup_frac=0.1):
    warm = max(1, int(total_steps * warmup_frac))

    def f(step):
        if step < warm:
            return step / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)
