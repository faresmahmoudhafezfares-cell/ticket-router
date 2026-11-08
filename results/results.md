Test set: 5550 messages (4550 in-scope, 1000 out-of-scope). Hardware: Intel(R) Xeon(R) Processor @ 2.10GHz, 2 logical CPU cores, CPU only, PyTorch 2.15.0+cu132, Linux. Latency = one message incl. preprocessing, mean of 200 runs.

| Model | Precision | In-scope accuracy | OOS recall | Routing accuracy | Size (MB) | Latency (ms) |
|---|---|---|---|---|---|---|
| 1 · tf-idf + Logistic Regression | FP64 | 79.2% | 90.9% | 82.7% | 40.4 | 0.6 |
| 2 · word2vec + Feedforward NN | FP32 | 79.2% | 81.6% | 81.2% | 2.8 | 0.1 |
| 3 · RoBERTa-base (fine-tuned) | FP32 | 91.6% | 88.6% | 91.6% | 502.7 | 51.6 |
| 3b · RoBERTa-base INT8 | INT8 | 90.3% | 90.5% | 90.8% | 245.8 | 21.0 |
| 4a · Distilled RoBERTa (4 layers) | FP32 | 87.6% | 90.9% | 88.7% | 275.8 | 18.8 |
| 4 · Distilled RoBERTa INT8 (deployed) | INT8 | 87.1% | 91.6% | 88.4% | 188.8 | 5.9 |
