"""Load CLINC150 plus the hand-written appliance examples, with fixed train/val/test splits."""
import csv
import json
import random
from pathlib import Path

from .preprocess import normalize

ROOT = Path(__file__).resolve().parent.parent
CLINC_PATH = ROOT / "data" / "data_full.json"
APPLIANCE_PATH = ROOT / "data" / "appliance_examples.csv"


APPLIANCE_OVERSAMPLE = 4   # 26 x 4 = 104 training rows per appliance intent, like CLINC's 100


def _appliance_splits(seed: int = 0):
    """Each appliance intent has 40 examples: 26 train / 4 val / 10 test."""
    by_intent = {}
    with open(APPLIANCE_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_intent.setdefault(row["intent"], []).append(row["text"])
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for intent, texts in sorted(by_intent.items()):
        texts = texts[:]
        rng.shuffle(texts)
        splits["train"] += [(t, intent) for t in texts[:26]]
        splits["val"] += [(t, intent) for t in texts[26:30]]
        splits["test"] += [(t, intent) for t in texts[30:40]]
    return splits


def load_splits(clean: bool = True):
    """Returns {'train': [(text, intent)], 'val': [...], 'test': [...]}.

    The out-of-scope examples are kept as the class 'oos'. The test set is the
    CLINC150 test set (4500 in-scope + 1000 out-of-scope) plus 50 appliance examples,
    and it is the same for every model. The 130 appliance training messages are repeated
    4 times (oversampling) so the 5 small appliance intents are not drowned out."""
    raw = json.load(open(CLINC_PATH, encoding="utf-8"))
    app = _appliance_splits()
    splits = {
        "train": raw["train"] + raw["oos_train"] + app["train"] * APPLIANCE_OVERSAMPLE,
        "val": raw["val"] + raw["oos_val"] + app["val"],
        "test": raw["test"] + raw["oos_test"] + app["test"],
    }
    splits = {k: [(t, l) for t, l in v] for k, v in splits.items()}
    if clean:
        splits = {k: [(normalize(t), l) for t, l in v] for k, v in splits.items()}
    return splits


def label_list(splits):
    return sorted({l for _, l in splits["train"]})
