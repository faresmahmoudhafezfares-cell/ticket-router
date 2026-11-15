"""Customer Support Ticket Router - the application.

Run:  python app.py      (opens http://127.0.0.1:7860 in the browser)

Pipeline for every message:
  1. regex normalization + model-code extraction + edit-distance spelling fix
  2. ELIZA-style reply if the whole message is small talk
  3. intent prediction by the deployed model (distilled RoBERTa, INT8)
  4. team, or "Human Agent" if out of scope / confidence below the threshold
The other models of the ladder are run on the same cleaned text and shown side by side.
"""
import json
import os
import time

import gradio as gr

from router.models import MODELS, available_models
from router.preprocess import normalize
from router.route import explain
from router.smalltalk import eliza_reply
from router.ui import HOW_IT_WORKS, result_html, results_html

DEPLOYED = "student_int8"                       # the quantized Transformer
models = available_models(include_teacher=os.environ.get("SKIP_TEACHER") != "1")
thresholds = json.load(open(MODELS / "thresholds.json")) if (MODELS / "thresholds.json").exists() else {}
deployed = {m.short: m for m in models}.get(DEPLOYED) or models[-1]
DEFAULT_THRESHOLD = thresholds.get(deployed.short, 0.5)
normalize("warm up")                            # loads the known-word list before the first message
for _m in models:                               # first call of each model is slow; do it now
    _m.predict_proba(["warm up"])


def run_model(model, clean, threshold):
    """Run one model on the cleaned text; return its decision and how long it took."""
    start = time.perf_counter()
    probs = model.predict_proba([clean])[0]
    ms = (time.perf_counter() - start) * 1000
    return explain(probs, model.labels, threshold), ms


def route(message, threshold=None):
    """Full pipeline for one message. Returns the result panel (HTML) and the comparison table."""
    if not message or not message.strip():
        return "<p>Please type a message.</p>", []
    threshold = DEFAULT_THRESHOLD if threshold is None else float(threshold)

    # 1. classical preprocessing
    t0 = time.perf_counter()
    clean, info = normalize(message, return_info=True)
    pre_ms = (time.perf_counter() - t0) * 1000

    # 3. every model predicts on the same cleaned text (the deployed one uses the slider threshold)
    rows, decision, model_ms = [], None, 0.0
    for m in models:
        is_deployed = m is deployed
        d, ms = run_model(m, clean, threshold if is_deployed else thresholds.get(m.short, 0.5))
        rows.append([m.name, m.precision, d["best"], f'{d["confidence"]:.2f}', d["team"], f"{ms:.1f}"])
        if is_deployed:
            decision, model_ms = d, ms

    # 2. small talk: a rule that covers the whole message answers first;
    #    otherwise the reflective rules answer when the model itself says "small talk"
    reply = eliza_reply(message)
    decision["rule_first"] = reply is not None
    if reply is None and decision["team"] == "Small Talk":
        reply = eliza_reply(message, standalone_only=False) or "Happy to chat! How can I help you today?"

    panel = result_html(clean=clean, info=info, reply=reply, decision=decision, threshold=threshold,
                        model_name=deployed.name, pre_ms=pre_ms, model_ms=model_ms)
    return panel, rows


EXAMPLES = [
    "My dishwahser DW-6534 is not draining!!!",
    "hello",
    "my washng machine WM 8120 shakes like crazy during spin",
    "my credit card was declined at the store",
    "can you book me a flight to Izmir next friday",
    "can you recommend a good movie for tonight",
    "refrigerator is too warm inside",
    "is my fridge still under warranty? bought it 2 years ago",
    "the oven won't heat and the clock is blinking",
]

with gr.Blocks(title="Support Ticket Router") as demo:
    gr.Markdown("# Customer Support Ticket Router\n"
                "Type a customer message. The app cleans it, predicts the intent and sends it to the "
                "right team. Unclear or out-of-scope messages go to a human agent.")
    with gr.Tab("Route a message"):
        with gr.Row():
            inp = gr.Textbox(label="Customer message", lines=2, scale=4,
                             placeholder="e.g. my dishwasher DW-6534 won't drain")
            btn = gr.Button("Route", variant="primary", scale=1)
        thr = gr.Slider(0.50, 0.99, value=DEFAULT_THRESHOLD, step=0.01,
                        label="Confidence needed to route without a human "
                              f"(tuned on the validation set: {DEFAULT_THRESHOLD:.2f})")
        gr.Examples(EXAMPLES, inputs=inp, label="Try one of these")
        out = gr.HTML()
        table = gr.Dataframe(headers=["Model", "Precision", "Top intent", "Confidence", "Team", "Latency (ms)"],
                             label="All models side by side (same cleaned text)", interactive=False)
        btn.click(route, [inp, thr], [out, table])
        inp.submit(route, [inp, thr], [out, table])
    with gr.Tab("Results"):
        gr.Markdown("### The four-model ladder on the same held-out test set")
        gr.HTML(results_html(DEPLOYED))
    with gr.Tab("How it works"):
        gr.Markdown(HOW_IT_WORKS)

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Default(primary_hue="orange", neutral_hue="slate"))
