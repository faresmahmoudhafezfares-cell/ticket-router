"""ELIZA-style rule-based replies for greetings and small talk (technique area B).

Like Weizenbaum's ELIZA, each rule is a regular expression plus a reply template;
parts of the user's message captured by the pattern can be reflected back.
"""
import random
import re

RULES = [
    (r"^(hi|hello|hey|good (morning|afternoon|evening)|merhaba|selam)\b",
     ["Hello! How can I help you with your appliance or account today?",
      "Hi there! Tell me what's going on and I'll route you to the right team."]),
    (r"\bhow are you\b",
     ["I'm doing well, thanks for asking! What can I help you with?"]),
    (r"\b(thank you|thanks|thx|teşekkürler|tesekkurler)\b",
     ["You're welcome! Anything else I can help with?"]),
    (r"^(bye|goodbye|see you|good night)\b",
     ["Goodbye! Have a great day."]),
    (r"\bare you (a )?(bot|robot|human|real)\b",
     ["I'm an automated assistant. If you need a person, just say so and I'll connect you."]),
    (r"\b(who|what) (made|built|created) you\b",
     ["I was built as a support-ticket router for a course project."]),
    (r"\bwhat('s| is) your name\b",
     ["I'm the support router bot."]),
    (r"\bi (feel|am feeling|am) (so )?(angry|upset|frustrated|annoyed|sad)\b",
     ["I'm sorry you feel {3}. Tell me what happened and I'll get you to the right team."]),
    (r"\bi need (.+)",
     ["Why do you need {1}? Tell me a bit more so I can route you correctly."]),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), r) for p, r in RULES]

# Rules that are pure small talk and may answer on their own (the last two rules
# only fire when the classifier also says the message is small talk).
STANDALONE = 7


def eliza_reply(text: str, rng=random.Random(0), standalone_only=True):
    """Return a reply if a small-talk rule matches, else None."""
    rules = _COMPILED[:STANDALONE] if standalone_only else _COMPILED
    t = text.strip()
    for pat, replies in rules:
        m = pat.search(t)
        if m:
            # On its own, a rule only answers when it covers (almost) the whole message,
            # so "hi my fridge is broken" is still routed to a team.
            rest = (t[:m.start()] + " " + t[m.end():]).split()
            if standalone_only and len(rest) > 2:
                return None
            groups = (m.group(0),) + m.groups()
            return rng.choice(replies).format(*[g or "" for g in groups])
    return None
