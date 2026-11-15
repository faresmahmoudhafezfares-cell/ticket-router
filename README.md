# Customer Support Ticket Router

SWE015 – Introduction to Large Language Models · Homework, Use Case 2

A customer types a free-text support message. The app cleans it, predicts the **intent** with a
confidence score and shows the **team** it is routed to. Low-confidence or out-of-scope messages
go to a **human agent**; greetings and small talk get a rule-based (ELIZA-style) reply. The
predictions of all four models are shown side by side, and the app's decision comes from the
**quantized Transformer**.

Domain detail changed from the scenario (allowed by the assignment): messages are in **English**.

## Setup and run

Needs Python 3.11–3.13 and about 2 GB of free disk space. CPU only, no GPU needed.

```bash
git clone https://github.com/faresmahmoudhafezfares-cell/ticket-router.git
cd ticket-router
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_models.py          # the four trained models, ~540 MB, from the GitHub Release
python app.py                      # opens http://127.0.0.1:7860
```

**Single run command:** `python app.py`

The trained weights are not in the repository. They are attached to the
[v1.1 release](https://github.com/faresmahmoudhafezfares-cell/ticket-router/releases/tag/v1.1)
and `download_models.py` unpacks them into `models/`.

## What the app shows

* **Route a message** – the outcome (team, human agent, or small-talk reply), the four pipeline
  steps for that message (cleaned text, model codes and typo fixes; small-talk check; top-3 intents
  of the deployed model; the decision against the confidence threshold), and the predictions of
  all models side by side with their latency. A slider changes the confidence threshold.
* **Results** – the benchmark table below.
* **How it works** – a plain-language description of the pipeline and the models.

## The four models

| # | Model | Technique area |
|---|---|---|
| 1 | tf-idf (word 1–2 grams) + logistic regression | B · classical NLP |
| 2 | word2vec trained on the training messages, averaged → feedforward network (100 → 512 → 156) trained with SGD + backpropagation | A · neural networks (+ B embeddings) |
| 3 | RoBERTa-base (BERT-family encoder, BPE tokenizer) fine-tuned with a classification head | C · Transformers |
| 4 | 4-layer student distilled from model 3, then INT8 quantization — **runs in the app** | D · efficient models |

Preprocessing (area B, same for every model): regex normalization, regex extraction of product
model codes (e.g. `DW-6534`), minimum-edit-distance correction of misspelled product names;
ELIZA-style regex rules reply to small talk.

**Distillation.** The student copies the teacher's embeddings, layers 2, 5, 8, 11 and classifier
head, then trains for 5 epochs on `0.5 × KL divergence to the teacher's softened predictions (T = 2) + 0.5 × cross-entropy`.

**Quantization.** PyTorch dynamic quantization (`router/models.py`, `_to_int8`): the weights of
every Linear layer are stored as 8-bit integers and their activations are quantized on the fly;
the word-embedding table (50,265 × 768) is stored as 8-bit integers with one scale per row.
LayerNorm, position embeddings and biases stay FP32. No calibration data is needed.

RoBERTa is used instead of BERT (same architecture and technique area, pretrained longer on more
text) because its pretrained weights could be obtained without the Hugging Face Hub; see Data.

## Results

Test set: 5550 messages (4550 in-scope, 1000 out-of-scope). Hardware: Intel(R) Xeon(R) Processor @ 2.30GHz, 2 logical CPU cores, CPU only, PyTorch 2.15.0+cu132, Linux. Latency = one message incl. preprocessing, mean of 200 runs.

| Model | Precision | In-scope accuracy | OOS recall | Routing accuracy | Size (MB) | Latency (ms) |
|---|---|---|---|---|---|---|
| 1 · tf-idf + Logistic Regression | FP64 | 79.2% | 90.9% | 82.7% | 40.4 | 0.4 |
| 2 · word2vec + Feedforward NN | FP32 | 79.2% | 81.6% | 81.2% | 2.8 | 0.1 |
| 3 · RoBERTa-base (fine-tuned) | FP32 | 91.6% | 88.6% | 91.6% | 502.7 | 46.7 |
| 3b · RoBERTa-base INT8 | INT8 | 88.9% | 91.1% | 89.7% | 130.4 | 17.9 |
| 4a · Distilled RoBERTa (4 layers) | FP32 | 87.6% | 90.9% | 88.7% | 275.8 | 15.8 |
| 4 · Distilled RoBERTa INT8 (deployed) | INT8 | 86.9% | 91.4% | 88.2% | 73.4 | 3.7 |

In-scope accuracy = right intent on the in-scope messages (a message wrongly sent to a human counts
as an error). OOS recall = share of out-of-scope messages sent to a human. Routing accuracy = right
team over all messages. Size = files on disk. Rows 3b and 4a are extra rows that separate the effect
of quantization from the effect of distillation. The confidence thresholds were tuned on the
validation set only (`models/thresholds.json`).

## Reproduce the results table

```bash
python download_models.py --all    # adds the extra models 3b and 4a
python evaluate.py                 # ~10 min on a laptop CPU; writes results/results.md and results.json
python tests/test_pipeline.py      # quick checks of regex, edit distance, small talk
```

## Retrain everything (optional; CPU is enough, the Transformers take about 2 hours)

```bash
python train_baseline.py                          # model 1
python train_ffn.py                               # model 2
python train_transformer.py --base roberta-base   # model 3
python distill_quantize.py                        # model 4 (distillation, then INT8)
python evaluate.py
```

## Project layout

```
app.py                  Gradio application
evaluate.py             evaluates every model on the same test set; size and latency benchmark
train_baseline.py       model 1        train_ffn.py          model 2
train_transformer.py    model 3        distill_quantize.py   model 4
download_models.py      fetches the trained models from the GitHub Release
router/preprocess.py    regex normalization, model codes, edit distance
router/smalltalk.py     ELIZA-style rules
router/route.py         confidence threshold -> team or human agent
router/teams.py         intent -> team mapping
router/models.py        one interface for all models; INT8 quantization and loading
router/ui.py            HTML for the app's result panel and results table
data/                   CLINC150 + 200 appliance examples
results/                results table and each model's test-set errors
tests/                  small checks for the classical NLP parts
tools/                  how the pretrained RoBERTa weights were obtained
```

## Data and sources

* **CLINC150** (Larson et al., 2019, "An Evaluation Dataset for Intent Classification and
  Out-of-Scope Prediction", EMNLP) — 150 intents in 10 domains + out-of-scope messages.
  <https://github.com/clinc/oos-eval> (`data_full.json`, CC BY 3.0).
* **Appliance examples** — 200 messages for 5 home-appliance intents written for this project
  (`data/appliance_examples.csv`), split 26 / 4 / 10 per intent into train / validation / test;
  the training messages are repeated 4× so each appliance intent has ~100 training rows, like CLINC.
* **Pretrained model** — RoBERTa-base (Liu et al., 2019). The weights were taken from the
  RoBERTa-base encoder inside spaCy's `en_core_web_trf` 3.8.0 package (MIT) and converted with
  `tools/roberta_from_spacy.py`; `--base roberta-base` (Hugging Face Hub) works too.

## AI usage

Claude (Anthropic) was used to write and debug the code, run the training, draft the 200 appliance
example messages, and draft the slides and the demo script.
