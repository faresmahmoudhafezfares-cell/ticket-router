"""HTML pieces for the Gradio app (presentation only, no model logic)."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CARD = ("border:1px solid var(--border-color-primary);border-radius:12px;padding:14px 16px;"
        "background:var(--background-fill-secondary);min-width:0")
MUTED = "color:var(--body-text-color-subdued);font-size:13px"
ACCENT = "var(--color-accent)"


def show(conf, threshold):
    """Confidence as text; one more decimal when it would otherwise look equal to the threshold."""
    return f"{conf:.3f}" if f"{conf:.2f}" == f"{threshold:.2f}" else f"{conf:.2f}"


def esc(text):
    return html.escape(str(text))


def chip(text):
    return (f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;font-size:13px;'
            f'border:1px solid var(--border-color-primary);margin:2px 4px 2px 0">{esc(text)}</span>')


def bar(fraction, color=ACCENT):
    width = max(0.0, min(1.0, fraction)) * 100
    return ('<div style="height:8px;border-radius:4px;background:var(--border-color-primary);overflow:hidden">'
            f'<div style="height:8px;width:{width:.1f}%;background:{color}"></div></div>')


def step(number, title, body):
    return (f'<div style="{CARD}"><div style="{MUTED};text-transform:uppercase;letter-spacing:.06em">'
            f'Step {number}</div><div style="font-weight:600;margin:2px 0 8px">{esc(title)}</div>{body}</div>')


def banner(label, headline, detail, strong=True):
    edge = ACCENT if strong else "var(--body-text-color-subdued)"
    return (f'<div style="{CARD};border-left:6px solid {edge};margin-bottom:12px">'
            f'<div style="{MUTED};text-transform:uppercase;letter-spacing:.06em">{esc(label)}</div>'
            f'<div style="font-size:26px;font-weight:700;line-height:1.25;margin:2px 0">{esc(headline)}</div>'
            f'<div style="font-size:14px">{detail}</div></div>')


def result_html(*, clean, info, reply, decision, threshold, model_name, pre_ms, model_ms):
    """The result panel: one banner with the outcome, then the four pipeline steps."""
    conf = decision["confidence"]
    if reply:
        top = banner("Small talk · rule-based reply", reply, "Answered by an ELIZA-style rule, no team needed.")
    elif decision["reason"] == "routed":
        top = banner("Routed to", decision["team"],
                     f'Intent <code>{esc(decision["best"])}</code> · confidence {conf:.2f}')
    elif decision["reason"] == "out_of_scope":
        top = banner("Sent to", "Human Agent",
                     "The model thinks this message is out of scope (not one of the supported intents).", False)
    else:
        top = banner("Sent to", "Human Agent",
                     f'Best guess <code>{esc(decision["best"])}</code>, but confidence {show(conf, threshold)} is below '
                     f'the {threshold:.2f} needed to route automatically.', False)

    codes = "".join(chip(c) for c in info["model_codes"]) or f'<span style="{MUTED}">none</span>'
    fixes = "".join(chip(f"{a} → {b}") for a, b in info["corrections"]) or f'<span style="{MUTED}">none</span>'
    s1 = step(1, "Clean the text",
              f'<div style="font-family:var(--font-mono);font-size:13px;word-break:break-word">{esc(clean)}</div>'
              f'<div style="{MUTED};margin-top:8px">Model codes (regex)</div>{codes}'
              f'<div style="{MUTED};margin-top:6px">Typos fixed (edit distance)</div>{fixes}')

    rule_fired = bool(reply) and decision.get("rule_first", False)
    s2 = step(2, "Small talk?",
              "Yes: a greeting / thanks rule matched, so the bot replies itself." if rule_fired
              else "No rule matched the whole message, so it goes to the model.")

    rows = "".join(
        f'<div style="display:flex;justify-content:space-between;font-size:13px;margin-top:6px">'
        f'<span style="font-family:var(--font-mono)">{esc(name)}</span><span>{p:.2f}</span></div>{bar(p)}'
        for name, p in decision["top"])
    s3 = step(3, "Predict the intent", f'<div style="{MUTED}">{esc(model_name)}</div>{rows}')

    if rule_fired:
        verdict = "The rule in step 2 already replied, so no team is needed."
    elif reply:
        verdict = "The model says this is small talk, so a rule-based reply is sent."
    elif decision["reason"] == "routed":
        verdict = f"{show(conf, threshold)} ≥ {threshold:.2f} → route to <b>{esc(decision['team'])}</b>"
    elif decision["reason"] == "out_of_scope":
        verdict = "Predicted out of scope → <b>Human Agent</b>"
    else:
        verdict = f"{show(conf, threshold)} &lt; {threshold:.2f} → <b>Human Agent</b>"
    s4 = step(4, "Decide", f'{verdict}<div style="{MUTED};margin-top:8px">'
                          f'Preprocessing {pre_ms:.1f} ms · model {model_ms:.1f} ms</div>')

    grid = ('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px">'
            f'{s1}{s2}{s3}{s4}</div>')
    return top + grid


def results_html(deployed_key="student_int8"):
    """The benchmark table, read from results/results.json (written by evaluate.py)."""
    path = ROOT / "results" / "results.json"
    if not path.exists():
        return "<p>No results yet. Run <code>python evaluate.py</code> first.</p>"
    data = json.load(open(path))
    rows = data["rows"]
    max_size = max(r["size_mb"] for r in rows)
    max_lat = max(r["latency_ms_mean"] for r in rows)
    th = 'style="text-align:left;padding:8px 10px;border-bottom:2px solid var(--border-color-primary)"'
    td = 'style="padding:8px 10px;border-bottom:1px solid var(--border-color-primary);vertical-align:middle"'
    body = ""
    for r in rows:
        mark = "font-weight:700" if r["key"] == deployed_key else ""
        body += (
            f'<tr style="{mark}"><td {td}>{esc(r["model"])}</td><td {td}>{esc(r["precision"])}</td>'
            f'<td {td}>{r["in_scope_accuracy"]*100:.1f}%{bar(r["in_scope_accuracy"])}</td>'
            f'<td {td}>{r["oos_recall"]*100:.1f}%{bar(r["oos_recall"])}</td>'
            f'<td {td}>{r["size_mb"]:.1f}{bar(r["size_mb"]/max_size, "var(--body-text-color-subdued)")}</td>'
            f'<td {td}>{r["latency_ms_mean"]:.1f}{bar(r["latency_ms_mean"]/max_lat, "var(--body-text-color-subdued)")}</td></tr>')
    return (
        '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:14px">'
        f'<tr><th {th}>Model</th><th {th}>Precision</th><th {th}>Accuracy (in-scope)</th>'
        f'<th {th}>Out-of-scope recall</th><th {th}>Size (MB)</th><th {th}>Latency (ms)</th></tr>{body}</table></div>'
        f'<p style="{MUTED};margin-top:10px">Every model was tested on the same {data["test_size"]:,} held-out '
        f'messages. Latency: one message including preprocessing, mean of {data["latency_runs"]} runs. '
        f'Hardware: {esc(data["hardware"])}. The bold row is the model this app runs.</p>')


HOW_IT_WORKS = """
### What this app does
A customer writes a message. The app decides **which team should answer it**: a smart receptionist.

### One message, four steps
1. **Clean the text** (classical NLP). Regular expressions lowercase the text, remove symbols and pull out
   product codes such as `DW-6534`. Misspelled product names are fixed with **minimum edit distance**
   (`dishwahser` → `dishwasher`).
2. **Small talk?** (classical NLP). If the whole message is a greeting or a thank-you, an **ELIZA-style
   rule** answers it. No model needed.
3. **Predict the intent** (Transformer). A small, compressed RoBERTa model scores 156 possible intents,
   e.g. `dishwasher_issue`, `card_declined`, `book_flight`.
4. **Decide.** Every intent belongs to one of 11 teams. If the model's confidence is below the threshold,
   or it predicts *out of scope*, the message goes to a **human agent**.

### The four models (same task, increasing sophistication)
| # | Model | In one sentence |
|---|---|---|
| 1 | tf-idf + logistic regression | Counts words (rare words weigh more) and learns which words go with which intent. |
| 2 | word2vec + feedforward network | Turns each word into a vector, averages them, and a small neural network (trained with SGD and backpropagation) picks the intent. |
| 3 | RoBERTa-base, fine-tuned | A 12-layer pretrained Transformer (BERT family) that reads the whole sentence in context. Most accurate, but big and slow. |
| 4 | Distilled RoBERTa, INT8 | A 4-layer "student" trained to imitate model 3 (**knowledge distillation**), then stored as 8-bit integers (**quantization**). This is the model the app runs. |

### Why compress the model?
Model 3 is the most accurate, but it is hundreds of megabytes and takes tens of milliseconds per message on a
CPU. Distillation removes 8 of the 12 layers; quantization stores the numbers in 8 bits instead of 32. The
result is several times smaller and faster for a few points of accuracy. The **Results** tab has the numbers.
"""
