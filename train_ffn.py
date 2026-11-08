"""Model 2 (technique area A + B): our own word2vec embeddings, averaged per message,
fed into a feedforward neural network trained with SGD and backpropagation.
"""
import json

import numpy as np
import torch
from gensim.models import Word2Vec

from router.data import label_list, load_splits
from router.ffn import FFN, sentence_vectors

torch.manual_seed(0)
splits = load_splits()
labels = label_list(splits)
l2i = {l: i for i, l in enumerate(labels)}

# 1) word2vec (skip-gram) trained only on our training messages
sentences = [t.split() for t, _ in splits["train"]]
w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1, epochs=40,
               workers=1, seed=0)
w2v.wv.save("models/word2vec.kv")


def xy(split):
    x = torch.tensor(sentence_vectors([t for t, _ in splits[split]], w2v.wv))
    y = torch.tensor([l2i[l] for _, l in splits[split]])
    return x, y


xtr, ytr = xy("train")
xva, yva = xy("val")

# 2) feedforward network: 100 -> 512 -> 156, ReLU, dropout
net = FFN(100, 512, len(labels))
opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()
best, best_state = 0.0, None
for epoch in range(150):
    net.train()
    perm = torch.randperm(len(xtr))
    for i in range(0, len(perm), 64):
        b = perm[i:i + 64]
        loss = loss_fn(net(xtr[b]), ytr[b])
        opt.zero_grad()
        loss.backward()        # backpropagation
        opt.step()             # stochastic gradient descent update
    net.eval()
    with torch.no_grad():
        acc = (net(xva).argmax(1) == yva).float().mean().item()
    if acc > best:
        best, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
    if epoch % 10 == 0:
        print(f"epoch {epoch} loss {loss.item():.3f} val acc {acc:.4f}", flush=True)
print(f"best validation accuracy {best:.4f}")
torch.save(best_state, "models/ffn.pt")
json.dump(labels, open("models/labels.json", "w"))
