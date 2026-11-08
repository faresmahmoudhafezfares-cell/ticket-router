# Customer Support Ticket Router

SWE015 – Introduction to Large Language Models · Homework, Use Case 2

A customer types a free-text support message. The app cleans it, predicts the **intent**
with a confidence score and routes it to the right **team**. Out-of-scope or low-confidence
messages go to a **human agent**; greetings and small talk get a rule-based (ELIZA-style) reply.
Four models of increasing sophistication are compared on the same test set, and the app runs
the quantized Transformer.

## Setup and run

Python 3.11–3.13.

```bash
git clone https://github.com/OWNER/REPO.git
cd REPO
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_models.py          # trained models (from the GitHub Release, ~450 MB)
python app.py                      # then open http://127.0.0.1:7860
```

**Single run command:** `python app.py`

To also show the big 12-layer RoBERTa teacher in the app, run `python download_models.py --teacher`.

## Reproduce the results table

```bash
python evaluate.py                 # writes results/results.md and results/results.json
```

## Retrain everything (optional, CPU is enough but the Transformer takes ~1 h)

```bash
python train_baseline.py           # model 1: tf-idf + logistic regression
python train_ffn.py                # model 2: word2vec + feedforward network (SGD)
python train_transformer.py --base roberta-base   # model 3: fine-tuned RoBERTa
python distill_quantize.py         # model 4: distilled 4-layer student + INT8 quantization
python evaluate.py
```

## Models (the "model ladder")

| # | Model | Technique area |
|---|---|---|
| 1 | tf-idf (word 1–2 grams) + logistic regression | B · classical NLP |
| 2 | word2vec trained on our data, averaged → feedforward network trained with SGD + backpropagation | A · neural networks (+ B embeddings) |
| 3 | RoBERTa-base (BERT-family encoder) fine-tuned with a classification head | C · Transformers |
| 4 | 4-layer student distilled from model 3, then dynamic INT8 quantization — **runs in the app** | D · efficient models |

Preprocessing (area B, used by every model): regex normalization, regex extraction of product
model codes (e.g. `DW-6534`), minimum-edit-distance correction of misspelled product names,
ELIZA-style regex replies for small talk.

## Results

Test set: 5550 messages (4550 in-scope, 1000 out-of-scope). Hardware: Intel(R) Xeon(R) Processor @ 2.10GHz, 2 logical CPU cores, CPU only, PyTorch 2.15.0+cu132, Linux. Latency = one message incl. preprocessing, mean of 200 runs.

| Model | Precision | In-scope accuracy | OOS recall | Routing accuracy | Size (MB) | Latency (ms) |
|---|---|---|---|---|---|---|
| 1 · tf-idf + Logistic Regression | FP64 | 79.2% | 90.9% | 82.7% | 40.4 | 0.6 |
| 2 · word2vec + Feedforward NN | FP32 | 79.2% | 81.6% | 81.2% | 2.8 | 0.1 |
| 3 · RoBERTa-base (fine-tuned) | FP32 | 91.6% | 88.6% | 91.6% | 502.7 | 51.6 |
| 3b · RoBERTa-base INT8 | INT8 | 90.3% | 90.5% | 90.8% | 245.8 | 21.0 |
| 4a · Distilled RoBERTa (4 layers) | FP32 | 87.6% | 90.9% | 88.7% | 275.8 | 18.8 |
| 4 · Distilled RoBERTa INT8 (deployed) | INT8 | 87.1% | 91.6% | 88.4% | 188.8 | 5.9 |

The app runs model 4 (distilled RoBERTa, INT8). Thresholds for "send to a human agent" were tuned on the validation set (`models/thresholds.json`).

## Project layout

```
app.py                  Gradio application
evaluate.py             evaluates all models on the same test set
train_*.py, distill_quantize.py   training scripts
router/preprocess.py    regex normalization, model codes, edit distance
router/smalltalk.py     ELIZA-style rules
router/teams.py         intent -> team mapping
router/models.py        common interface for the models, INT8 loading
data/                   CLINC150 + 200 appliance examples
results/                results table and error lists
```

## Data

* **CLINC150** (Larson et al., 2019, "An Evaluation Dataset for Intent Classification and
  Out-of-Scope Prediction", EMNLP) — 150 intents in 10 domains + out-of-scope messages.
  <https://github.com/clinc/oos-eval> (`data_full.json`, CC BY 3.0).
* **Appliance examples** — 200 messages for 5 home-appliance intents written for this project
  (`data/appliance_examples.csv`), split 26 / 4 / 10 per intent into train / validation / test;
  the training messages are repeated 4× so each appliance intent has ~100 training rows, like CLINC.
* **Pretrained model** — RoBERTa-base (Liu et al., 2019). The weights used here were taken from the
  RoBERTa-base encoder inside spaCy's `en_core_web_trf` 3.8.0 package (MIT) and converted with
  `tools/roberta_from_spacy.py`; `--base roberta-base` (Hugging Face Hub) works too.

## AI usage

Claude (Anthropic) was used to help write and debug the code, draft the appliance example
messages, and draft the slides and demo script.
