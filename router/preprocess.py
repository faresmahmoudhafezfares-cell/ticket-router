"""Classical NLP preprocessing (technique area B).

1. Regular expressions: normalize text and pull out product model codes (e.g. DW-6534).
2. Minimum edit distance (Levenshtein, dynamic programming): fix misspelled product
   names such as "dishwahser" -> "dishwasher".
"""
import json
import re
import unicodedata
from pathlib import Path

# Product model codes look like DW-6534, WM 8120, RF2290: 2-3 letters, optional dash/space, 3-5 digits.
MODEL_CODE_RE = re.compile(r"\b([A-Za-z]{2,3})[-\s]?(\d{3,5})\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
REPEAT_RE = re.compile(r"(.)\1{2,}")            # "pleeeease" -> "pleease"
PUNCT_RE = re.compile(r"[^\w\s']")
SPACE_RE = re.compile(r"\s+")

# Product names the edit-distance corrector knows about.
PRODUCT_LEXICON = [
    "dishwasher", "washing", "machine", "washer", "dryer", "refrigerator", "fridge",
    "freezer", "oven", "microwave", "cooktop", "stove", "hob", "warranty", "guarantee",
    "appliance", "technician",
]

# Words we already know are real words (every word in the CLINC150 training messages).
# Only unknown words are spell-corrected, so "weather" is never turned into "washer".
_KNOWN_WORDS_PATH = Path(__file__).resolve().parent.parent / "data" / "data_full.json"
_known_words = None


def known_words() -> set:
    global _known_words
    if _known_words is None:
        raw = json.load(open(_KNOWN_WORDS_PATH, encoding="utf-8"))
        _known_words = {w for t, _ in raw["train"] + raw["oos_train"]
                        for w in re.findall(r"[a-z']+", t.lower())}
    return _known_words


def edit_distance(a: str, b: str) -> int:
    """Minimum edit distance (insert, delete, substitute all cost 1), computed with
    the classic dynamic-programming table, keeping only two rows."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1,                 # delete
                         cur[j - 1] + 1,              # insert
                         prev[j - 1] + (ca != cb))    # substitute (free if same letter)
        prev = cur
    return prev[-1]


def correct_word(word: str, lexicon=PRODUCT_LEXICON) -> str:
    """Replace a word with the closest product name if it is a near miss."""
    if len(word) < 4 or word in lexicon or not word.isalpha() or word in known_words():
        return word
    max_dist = 1 if len(word) <= 7 else 2
    best, best_d = word, max_dist + 1
    for cand in lexicon:
        if abs(len(cand) - len(word)) > max_dist:
            continue
        d = edit_distance(word, cand)
        if d < best_d:
            best, best_d = cand, d
    return best if best_d <= max_dist else word


def extract_model_codes(text: str) -> list[str]:
    """Return product model codes in a standard form, e.g. 'dw 6534' -> 'DW-6534'."""
    return [f"{a.upper()}-{n}" for a, n in MODEL_CODE_RE.findall(text)]


def normalize(text: str, return_info: bool = False):
    """Full preprocessing used before every model.

    Returns the cleaned text, and (optionally) a dict with the model codes found and
    the spelling corrections made, so the app can show them."""
    original = text
    text = unicodedata.normalize("NFKC", text)
    codes = extract_model_codes(text)
    text = MODEL_CODE_RE.sub(" ", text)          # remove codes; they do not decide the team
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = text.lower()
    text = REPEAT_RE.sub(r"\1\1", text)
    text = PUNCT_RE.sub(" ", text)
    words = SPACE_RE.sub(" ", text).strip().split()
    fixed, corrections = [], []
    for w in words:
        c = correct_word(w)
        if c != w:
            corrections.append((w, c))
        fixed.append(c)
    clean = " ".join(fixed)
    if return_info:
        return clean, {"original": original, "model_codes": codes, "corrections": corrections}
    return clean
