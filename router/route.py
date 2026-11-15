"""Turn a model's probabilities into a routing decision."""
import numpy as np

from .teams import HUMAN_AGENT, OOS, team_for


def decide(prob_row, labels, threshold):
    """Pick the most likely intent. If it is 'oos' or the model is not confident enough,
    send the message to a human agent."""
    i = int(np.argmax(prob_row))
    intent, conf = labels[i], float(prob_row[i])
    if intent == OOS or conf < threshold:
        return OOS, conf, HUMAN_AGENT
    return intent, conf, team_for(intent)


def predictions(probs, labels, threshold):
    return [decide(row, labels, threshold)[0] for row in probs]


def explain(prob_row, labels, threshold, top_k=3):
    """Same decision as decide(), plus the details the app shows: the top-k intents and
    why the message was routed or handed to a human."""
    order = np.argsort(prob_row)[::-1][:top_k]
    top = [(str(labels[i]), float(prob_row[i])) for i in order]
    best, conf = top[0]
    if best == OOS:
        reason, team = "out_of_scope", HUMAN_AGENT
    elif conf < threshold:
        reason, team = "low_confidence", HUMAN_AGENT
    else:
        reason, team = "routed", team_for(best)
    return {"top": top, "best": best, "confidence": conf, "team": team, "reason": reason}
