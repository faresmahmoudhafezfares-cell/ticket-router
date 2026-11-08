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
