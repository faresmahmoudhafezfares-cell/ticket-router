"""One common interface for the four models, used by evaluate.py and app.py."""
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def _size_mb(paths):
    total = 0
    for p in paths:
        p = Path(p)
        if p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        elif p.exists():
            total += p.stat().st_size
    return total / 1e6


class TfidfModel:
    name, short, precision = "1 · tf-idf + Logistic Regression", "tfidf", "FP64"

    def __init__(self):
        import joblib
        self.path = MODELS / "tfidf_logreg.joblib"
        self.model = joblib.load(self.path)
        self.labels = list(self.model.classes_)

    def predict_proba(self, texts):
        return self.model.predict_proba(texts)

    def size_mb(self):
        return _size_mb([self.path])


class FFNModel:
    name, short, precision = "2 · word2vec + Feedforward NN", "ffn", "FP32"

    def __init__(self):
        from gensim.models import KeyedVectors
        from .ffn import FFN
        self.labels = json.load(open(MODELS / "labels.json"))
        self.wv = KeyedVectors.load(str(MODELS / "word2vec.kv"))
        self.net = FFN(self.wv.vector_size, 512, len(self.labels))
        self.net.load_state_dict(torch.load(MODELS / "ffn.pt", weights_only=True))
        self.net.eval()

    @torch.no_grad()
    def predict_proba(self, texts):
        from .ffn import sentence_vectors
        x = torch.tensor(sentence_vectors(texts, self.wv))
        return torch.softmax(self.net(x), -1).numpy()

    def size_mb(self):
        return _size_mb([p for p in MODELS.iterdir() if p.name.startswith(("word2vec", "ffn"))])


class TransformerModel:
    def __init__(self, folder, name, short, precision="FP32"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.folder = MODELS / folder
        self.name, self.short, self.precision = name, short, precision
        self.tok = AutoTokenizer.from_pretrained(self.folder)
        if precision == "INT8":
            self.model = load_quantized(self.folder)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.folder).eval()
        cfg = self.model.config
        self.labels = [cfg.id2label[i] for i in range(cfg.num_labels)]

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=128):
        from .transformer import MAX_LEN
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                           max_length=MAX_LEN, return_tensors="pt")
            out.append(torch.softmax(self.model(**enc).logits.float(), -1).numpy())
        return np.concatenate(out)

    def size_mb(self):
        return _size_mb([self.folder])


def quantize_and_save(src, dst):
    """Dynamic INT8 quantization: the weights of every Linear layer are stored as 8-bit
    integers (with one scale per tensor); activations are quantized on the fly."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    q = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    os.makedirs(dst, exist_ok=True)
    torch.save(q.state_dict(), Path(dst) / "model_int8.pt")
    model.config.save_pretrained(dst)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)
    for extra in ("train_info.json",):
        if (Path(src) / extra).exists():
            (Path(dst) / extra).write_text((Path(src) / extra).read_text())


def load_quantized(folder):
    from transformers import AutoConfig, AutoModelForSequenceClassification
    cfg = AutoConfig.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_config(cfg).eval()
    q = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    q.load_state_dict(torch.load(Path(folder) / "model_int8.pt", weights_only=False))
    return q.eval()


def available_models(include_teacher=True):
    """Load every model whose files exist (the app still works if some are missing)."""
    out = []
    specs = [
        ("tfidf", lambda: TfidfModel()),
        ("ffn", lambda: FFNModel()),
        ("teacher", lambda: TransformerModel("roberta_teacher", "3 · RoBERTa-base (fine-tuned)", "teacher")),
        ("teacher_int8", lambda: TransformerModel("roberta_teacher_int8", "3b · RoBERTa-base INT8", "teacher_int8", "INT8")),
        ("student", lambda: TransformerModel("roberta_student", "4a · Distilled RoBERTa (4 layers)", "student")),
        ("student_int8", lambda: TransformerModel("roberta_student_int8", "4 · Distilled RoBERTa INT8 (deployed)", "student_int8", "INT8")),
    ]
    needed = {"tfidf": "tfidf_logreg.joblib", "ffn": "ffn.pt", "teacher": "roberta_teacher",
              "teacher_int8": "roberta_teacher_int8", "student": "roberta_student",
              "student_int8": "roberta_student_int8"}
    for key, make in specs:
        if not include_teacher and key.startswith("teacher"):
            continue
        if (MODELS / needed[key]).exists():
            out.append(make())
    return out
