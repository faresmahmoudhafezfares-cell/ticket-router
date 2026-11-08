"""Optional: build a Hugging Face RoBERTa-base folder from spaCy's en_core_web_trf package.

This is how the pretrained encoder for model 3 was obtained (the RoBERTa-base encoder
inside spaCy's en_core_web_trf 3.8.0, MIT licence). If you can reach the Hugging Face Hub
you can skip this and use `python train_transformer.py --base roberta-base` instead.

Usage:
  pip install srsly
  # download en_core_web_trf-3.8.0-py3-none-any.whl from
  # https://github.com/explosion/spacy-models/releases/tag/en_core_web_trf-3.8.0 and unzip it
  python tools/roberta_from_spacy.py <unzipped>/en_core_web_trf/en_core_web_trf-3.8.0/transformer/model pretrained/roberta-base
  python train_transformer.py --base pretrained/roberta-base
"""
import io
import json
import sys
from pathlib import Path

import srsly
import torch
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast

src, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

# 1. read spaCy's serialized model (msgpack); weights_only=True executes no pickled code
m = srsly.msgpack_loads(open(src, "rb").read())
shim = srsly.msgpack_loads(m["shims"][6][0])
sd = torch.load(io.BytesIO(shim["state"]), map_location="cpu", weights_only=True)
bpe = srsly.msgpack_loads(m["attrs"][3]["byte_bpe_processor"])

# 2. tokenizer (byte-level BPE vocabulary + merges)
json.dump(bpe["vocab"], open(out / "vocab.json", "w"), ensure_ascii=False)
with open(out / "merges.txt", "w", encoding="utf-8") as f:
    f.write("#version: 0.2\n")
    for a, b in bpe["merges"]:
        f.write(f"{a} {b}\n")
bt = ByteLevelBPETokenizer(str(out / "vocab.json"), str(out / "merges.txt"))
bt.post_processor = RobertaProcessing(("</s>", 2), ("<s>", 0))
tok = RobertaTokenizerFast(tokenizer_object=bt._tokenizer, bos_token="<s>", eos_token="</s>",
                           sep_token="</s>", cls_token="<s>", unk_token="<unk>", pad_token="<pad>",
                           mask_token="<mask>", model_max_length=512)

# 3. rename the weights to the Hugging Face layout (split the fused query/key/value matrix)
cfg = RobertaConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
                    intermediate_size=3072, max_position_embeddings=514, type_vocab_size=1,
                    layer_norm_eps=1e-5, pad_token_id=1, bos_token_id=0, eos_token_id=2)
model = RobertaModel(cfg, add_pooling_layer=False)
p = "curated_encoder."
new = {
    "embeddings.word_embeddings.weight": sd[p + "embeddings.inner.word_embeddings.weight"],
    "embeddings.token_type_embeddings.weight": sd[p + "embeddings.inner.token_type_embeddings.weight"],
    "embeddings.position_embeddings.weight": sd[p + "embeddings.inner.position_embeddings.weight"],
    "embeddings.LayerNorm.weight": sd[p + "embeddings.inner.layer_norm.weight"],
    "embeddings.LayerNorm.bias": sd[p + "embeddings.inner.layer_norm.bias"],
}
for i in range(12):
    s, t = f"{p}layers.{i}.", f"encoder.layer.{i}."
    qw, kw, vw = sd[s + "mha.input.weight"].chunk(3, 0)
    qb, kb, vb = sd[s + "mha.input.bias"].chunk(3, 0)
    for n, w, b in (("query", qw, qb), ("key", kw, kb), ("value", vw, vb)):
        new[t + f"attention.self.{n}.weight"], new[t + f"attention.self.{n}.bias"] = w, b
    pairs = {
        "attention.output.dense": "mha.output", "attention.output.LayerNorm": "attn_output_layernorm",
        "intermediate.dense": "ffn.intermediate", "output.dense": "ffn.output",
        "output.LayerNorm": "ffn_output_layernorm",
    }
    for hf, cur in pairs.items():
        new[t + hf + ".weight"] = sd[s + cur + ".weight"]
        new[t + hf + ".bias"] = sd[s + cur + ".bias"]
missing, unexpected = model.load_state_dict(new, strict=True), None
model.save_pretrained(out)
tok.save_pretrained(out)
for f in ("vocab.json", "merges.txt"):
    (out / f).unlink()
print("saved", out, "| check:", tok("Hello world")["input_ids"], "should be [0, 31414, 232, 2]")
