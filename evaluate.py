"""Evaluate every model on the SAME held-out test set and write the results table.

Usage:  python evaluate.py            (writes results/results.md and results/results.json)

For each model:
  * the confidence threshold for "send to a human agent" is chosen on the VALIDATION set
  * accuracy (in-scope), out-of-scope recall and routing accuracy are measured on the TEST set
  * size = files on disk (MB); latency = one message at a time, averaged over 200 runs
"""
import json
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from router.data import load_splits
from router.models import available_models
from router.preprocess import normalize
from router.route import predictions
from router.teams import OOS, team_for

N_LATENCY = 200
Path("results").mkdir(exist_ok=True)


def hardware():
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    import os
    return f"{cpu}, {os.cpu_count()} logical CPU cores, CPU only, PyTorch {torch.__version__}, {platform.system()}"


def scores(pred, gold):
    ins = [(p, g) for p, g in zip(pred, gold) if g != OOS]
    oos = [(p, g) for p, g in zip(pred, gold) if g == OOS]
    return {
        "in_scope_accuracy": sum(p == g for p, g in ins) / len(ins),
        "oos_recall": sum(p == OOS for p, _ in oos) / len(oos),
        "overall_accuracy": sum(p == g for p, g in zip(pred, gold)) / len(gold),
        "routing_accuracy": sum(team_for(p) == team_for(g) for p, g in zip(pred, gold)) / len(gold),
    }


def pick_threshold(probs, labels, gold):
    """Best threshold on validation = highest average of in-scope accuracy and OOS recall."""
    best_t, best_s = 0.0, -1
    for t in np.arange(0.0, 0.96, 0.01):
        s = scores(predictions(probs, labels, t), gold)
        val = (s["in_scope_accuracy"] + s["oos_recall"]) / 2
        if val > best_s:
            best_t, best_s = float(round(t, 2)), val
    return best_t


def latency_ms(model, raw_texts):
    for t in raw_texts[:10]:                      # warm-up
        model.predict_proba([normalize(t)])
    times = []
    for t in raw_texts[:N_LATENCY]:
        start = time.perf_counter()
        model.predict_proba([normalize(t)])       # preprocessing + model, one message
        times.append((time.perf_counter() - start) * 1000)
    return statistics.mean(times), statistics.median(times)


def main():
    splits = load_splits()
    raw_test = [t for t, _ in load_splits(clean=False)["test"]]
    rng = random.Random(0)
    lat_texts = rng.sample(raw_test, N_LATENCY)
    va_x, va_y = [t for t, _ in splits["val"]], [l for _, l in splits["val"]]
    te_x, te_y = [t for t, _ in splits["test"]], [l for _, l in splits["test"]]

    rows, thresholds = [], {}
    for m in available_models():
        print("evaluating", m.name, flush=True)
        thr = pick_threshold(m.predict_proba(va_x), m.labels, va_y)
        thresholds[m.short] = thr
        te_pred = predictions(m.predict_proba(te_x), m.labels, thr)
        s = scores(te_pred, te_y)
        mean_ms, med_ms = latency_ms(m, lat_texts)
        rows.append({"model": m.name, "key": m.short, "precision": m.precision, "threshold": thr,
                     **s, "size_mb": m.size_mb(), "latency_ms_mean": mean_ms,
                     "latency_ms_median": med_ms})
        print(rows[-1], flush=True)
        # keep the mistakes so we can show failure examples
        with open(f"results/errors_{m.short}.tsv", "w", encoding="utf-8") as f:
            f.write("text\ttrue\tpredicted\n")
            for x, g, p in zip(te_x, te_y, te_pred):
                if g != p:
                    f.write(f"{x}\t{g}\t{p}\n")

    hw = hardware()
    json.dump({"hardware": hw, "test_size": len(te_y), "latency_runs": N_LATENCY, "rows": rows},
              open("results/results.json", "w"), indent=2)
    json.dump(thresholds, open("models/thresholds.json", "w"), indent=2)
    lines = [
        f"Test set: {len(te_y)} messages ({sum(y != OOS for y in te_y)} in-scope, "
        f"{sum(y == OOS for y in te_y)} out-of-scope). Hardware: {hw}. "
        f"Latency = one message incl. preprocessing, mean of {N_LATENCY} runs.", "",
        "| Model | Precision | In-scope accuracy | OOS recall | Routing accuracy | Size (MB) | Latency (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['precision']} | {r['in_scope_accuracy']*100:.1f}% | "
                     f"{r['oos_recall']*100:.1f}% | {r['routing_accuracy']*100:.1f}% | "
                     f"{r['size_mb']:.1f} | {r['latency_ms_mean']:.1f} |")
    Path("results/results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
