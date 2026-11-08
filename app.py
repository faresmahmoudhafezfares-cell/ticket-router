"""Customer Support Ticket Router - the application.

Run:  python app.py      then open http://127.0.0.1:7860

Pipeline for every message:
  regex normalization + model-code extraction + edit-distance spelling fix
  -> ELIZA-style reply if it is just small talk
  -> intent prediction by the deployed model (distilled RoBERTa, INT8)
  -> team, or "Human Agent" if out-of-scope / low confidence
All other models are shown side by side for comparison.
"""
import json
import os
import time

import gradio as gr

from router.models import MODELS, available_models
from router.preprocess import normalize
from router.route import decide
from router.smalltalk import eliza_reply

DEPLOYED = "student_int8"
models = available_models(include_teacher=os.environ.get("SKIP_TEACHER") != "1")
thresholds = json.load(open(MODELS / "thresholds.json")) if (MODELS / "thresholds.json").exists() else {}
by_key = {m.short: m for m in models}
deployed = by_key.get(DEPLOYED) or models[-1]
normalize("warm up")          # loads the known-word list once, before the first message


def run_model(m, clean):
    start = time.perf_counter()
    probs = m.predict_proba([clean])[0]
    ms = (time.perf_counter() - start) * 1000
    intent, conf, team = decide(probs, m.labels, thresholds.get(m.short, 0.5))
    return intent, conf, team, ms


def route(message):
    if not message or not message.strip():
        return "Please type a message.", []
    t0 = time.perf_counter()
    clean, info = normalize(message, return_info=True)
    pre_ms = (time.perf_counter() - t0) * 1000

    rows = []
    for m in models:
        intent, conf, team, ms = run_model(m, clean)
        rows.append([m.name, m.precision, str(intent), f"{conf:.2f}", team, f"{ms:.1f}"])
    intent, conf, team, ms = run_model(deployed, clean)

    reply = eliza_reply(message)
    if reply is None and team == "Small Talk":
        reply = eliza_reply(message, standalone_only=False) or "Happy to chat! How can I help you today?"

    lines = ["### Result (deployed model: distilled RoBERTa, INT8)"]
    if reply:
        lines += [f"**Small talk detected - rule-based reply:** {reply}"]
    else:
        lines += [f"**Routed to:** `{team}`", "",
                  f"**Intent:** `{intent}`  ·  **Confidence:** {conf:.2f}  "
                  f"(threshold {thresholds.get(deployed.short, 0.5):.2f})"]
        if team == "Human Agent":
            lines += ["_The model is not confident or the message is out of scope, "
                      "so a human agent takes over._"]
    lines += ["", "#### Preprocessing",
              f"- Normalized text: `{clean}`",
              f"- Model codes found: {', '.join(info['model_codes']) or 'none'}",
              f"- Spelling fixes (edit distance): "
              f"{', '.join(f'{a} → {b}' for a, b in info['corrections']) or 'none'}",
              f"- Time: preprocessing {pre_ms:.1f} ms, deployed model {ms:.1f} ms"]
    return "\n".join(lines), rows


EXAMPLES = [
    "My dishwahser DW-6534 is not draining!!!",
    "hello",
    "my washng machine WM 8120 shakes like crazy during spin",
    "my credit card was declined at the store",
    "can you book me a flight to Izmir next friday",
    "what is the meaning of the color blue in poetry",
    "is my fridge still under warranty? bought it 2 years ago",
    "the oven won't heat and the clock is blinking",
]

with gr.Blocks(title="Support Ticket Router") as demo:
    gr.Markdown("# Customer Support Ticket Router\n"
                "Type a customer message. The app cleans it, predicts the intent, and sends it "
                "to the right team. Unclear or out-of-scope messages go to a human agent.")
    with gr.Row():
        inp = gr.Textbox(label="Customer message", lines=2, scale=4,
                         placeholder="e.g. my dishwasher DW-6534 won't drain")
        btn = gr.Button("Route", variant="primary", scale=1)
    out = gr.Markdown()
    table = gr.Dataframe(headers=["Model", "Precision", "Intent", "Confidence", "Team", "Latency (ms)"],
                         label="All models side by side", interactive=False)
    gr.Examples(EXAMPLES, inputs=inp)
    btn.click(route, inp, [out, table])
    inp.submit(route, inp, [out, table])

if __name__ == "__main__":
    demo.launch()
