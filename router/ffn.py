import numpy as np
import torch


class FFN(torch.nn.Module):
    """Feedforward network: input -> hidden (ReLU, dropout) -> one score per intent."""

    def __init__(self, d_in, d_hidden, n_out):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_hidden), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(d_hidden, n_out))

    def forward(self, x):
        return self.net(x)


def sentence_vectors(texts, wv):
    """Average the word2vec vectors of the known words in each message."""
    out = np.zeros((len(texts), wv.vector_size), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs = [wv[w] for w in t.split() if w in wv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out
