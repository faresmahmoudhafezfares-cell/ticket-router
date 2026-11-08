"""Model 3 (technique area C): fine-tune a pretrained RoBERTa encoder (a BERT-family
Transformer) with a classification head on top, for 156 intents (150 CLINC + 5 appliance + oos).

Usage:  python train_transformer.py --base roberta-base --epochs 3
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from router.data import label_list, load_splits
from router.transformer import batches, linear_schedule, log, predict_logits

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="roberta-base", help="pretrained model name or local folder")
ap.add_argument("--out", default="models/roberta_teacher")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--max_steps", type=int, default=0, help="for a quick test run only")
args = ap.parse_args()

torch.manual_seed(0)
torch.set_num_threads(max(1, torch.get_num_threads()))
splits = load_splits()
labels = label_list(splits)
l2i = {l: i for i, l in enumerate(labels)}
tr_x, tr_y = [t for t, _ in splits["train"]], [l2i[l] for _, l in splits["train"]]
va_x, va_y = [t for t, _ in splits["val"]], [l2i[l] for _, l in splits["val"]]

tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForSequenceClassification.from_pretrained(
    args.base, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
steps_per_epoch = (len(tr_x) + args.batch_size - 1) // args.batch_size
total = args.max_steps or steps_per_epoch * args.epochs
sched = linear_schedule(opt, total)
loss_fn = torch.nn.CrossEntropyLoss()

step, best = 0, -1.0
for epoch in range(args.epochs):
    model.train()
    for b in batches(tr_x, tr_y, tok, args.batch_size, shuffle=True, seed=epoch):
        loss = loss_fn(model(**b["enc"]).logits, b["labels"])
        loss.backward()                       # backpropagation through all 12 layers
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        step += 1
        if step % 25 == 0:
            log(f"epoch {epoch+1} step {step}/{total} loss {loss.item():.4f}")
        if args.max_steps and step >= args.max_steps:
            break
    va_pred = predict_logits(model, tok, va_x).argmax(1).tolist()
    acc = sum(p == y for p, y in zip(va_pred, va_y)) / len(va_y)
    log(f"epoch {epoch+1} validation accuracy {acc:.4f}")
    if acc > best:
        best = acc
        model.save_pretrained(args.out); tok.save_pretrained(args.out)
        json.dump({"epoch": epoch + 1, "val_acc": acc}, open(Path(args.out) / "train_info.json", "w"))
    if args.max_steps and step >= args.max_steps:
        break
log(f"done, best validation accuracy {best:.4f}")
