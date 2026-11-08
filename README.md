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
python download_models.py          # trained models (from the GitHub Release, ~MODELS_SIZE)
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

RESULTS_TABLE

## Project layout

```
app.py                  Gradio application
evaluate.py             evaluates all models on the same test set
train_*.py, distill_quantize.py   training scripts
router/preprocess.py    regex normalization, model codes, edit distance
router/smalltalk.py     ELIZA-style rules
router/teams.py         intent -> team mapping
router/models.py        common interface for the models, INT8 loading
data/                   CLINC150 + 120 appliance examples
results/                results table and error lists
```

## Data

* **CLINC150** (Larson et al., 2019, "An Evaluation Dataset for Intent Classification and
  Out-of-Scope Prediction", EMNLP) — 150 intents in 10 domains + out-of-scope messages.
  <https://github.com/clinc/oos-eval> (`data_full.json`, CC BY 3.0).
* **Appliance examples** — 120 messages for 5 home-appliance intents written for this project
  (`data/appliance_examples.csv`), split 14 / 4 / 6 per intent into train / validation / test.
* **Pretrained model** — RoBERTa-base (Liu et al., 2019).

## AI usage

Claude (Anthropic) was used to help write and debug the code, draft the appliance example
messages, and draft the slides.
