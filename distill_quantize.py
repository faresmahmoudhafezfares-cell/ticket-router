"""Model 4 (technique area D): knowledge distillation + INT8 quantization.

1. Build a 4-layer student by copying the teacher's embeddings, 4 of its 12 layers
   (2, 5, 8, 11) and its classification head.
2. Train the student to copy the teacher's soft predictions (distillation loss) and the
   true labels (cross-entropy).
3. Quantize the student's Linear layers to 8-bit integers with PyTorch dynamic quantization.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from router.data import load_splits
from router.transformer import batches, linear_schedule, log, predict_logits

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default="models/roberta_teacher")
ap.add_argument("--out", default="models/roberta_student")
ap.add_argument("--layers", default="2,5,8,11")
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--T", type=float, default=2.0, help="softmax temperature")
ap.add_argument("--alpha", type=float, default=0.5, help="weight of the distillation loss")
args = ap.parse_args()
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(args.teacher)
teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher).eval()
l2i = teacher.config.label2id
splits = load_splits()
tr_x, tr_y = [t for t, _ in splits["train"]], [l2i[l] for _, l in splits["train"]]
va_x, va_y = [t for t, _ in splits["val"]], [l2i[l] for _, l in splits["val"]]

# --- 1. student = smaller copy of the teacher
keep = [int(i) for i in args.layers.split(",")]
cfg = copy.deepcopy(teacher.config)
cfg.num_hidden_layers = len(keep)
student = AutoModelForSequenceClassification.from_config(cfg)
student.roberta.embeddings.load_state_dict(teacher.roberta.embeddings.state_dict())
for new_i, old_i in enumerate(keep):
    student.roberta.encoder.layer[new_i].load_state_dict(teacher.roberta.encoder.layer[old_i].state_dict())
student.classifier.load_state_dict(teacher.classifier.state_dict())

# --- 2. distillation
log("computing teacher predictions on the training set")
t_logits = predict_logits(teacher, tok, tr_x)
del teacher
opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
total = ((len(tr_x) + args.batch_size - 1) // args.batch_size) * args.epochs
sched = linear_schedule(opt, total)
step, best = 0, -1.0
for epoch in range(args.epochs):
    student.train()
    for b in batches(tr_x, tr_y, tok, args.batch_size, shuffle=True, seed=100 + epoch, extra=t_logits):
        s = student(**b["enc"]).logits
        kd = F.kl_div(F.log_softmax(s / args.T, -1), F.softmax(b["extra"] / args.T, -1),
                      reduction="batchmean") * args.T ** 2
        ce = F.cross_entropy(s, b["labels"])
        loss = args.alpha * kd + (1 - args.alpha) * ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        step += 1
        if step % 50 == 0:
            log(f"epoch {epoch+1} step {step}/{total} loss {loss.item():.4f}")
    pred = predict_logits(student, tok, va_x).argmax(1).tolist()
    acc = sum(p == y for p, y in zip(pred, va_y)) / len(va_y)
    log(f"epoch {epoch+1} validation accuracy {acc:.4f}")
    if acc > best:
        best = acc
        student.save_pretrained(args.out); tok.save_pretrained(args.out)
        json.dump({"epoch": epoch + 1, "val_acc": acc, "layers": keep},
                  open(Path(args.out) / "train_info.json", "w"))

# --- 3. INT8 dynamic quantization of the best student
from router.models import quantize_and_save  # noqa: E402

quantize_and_save(args.out, "models/roberta_student_int8")
log(f"done, best student validation accuracy {best:.4f}")
