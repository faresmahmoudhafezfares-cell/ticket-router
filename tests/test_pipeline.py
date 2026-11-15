"""Small checks for the classical NLP parts. Run:  python tests/test_pipeline.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.preprocess import edit_distance, extract_model_codes, normalize  # noqa: E402
from router.smalltalk import eliza_reply  # noqa: E402
from router.teams import HUMAN_AGENT, team_for  # noqa: E402


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("dishwahser", "dishwasher") == 2
    assert edit_distance("oven", "oven") == 0
    assert edit_distance("", "abc") == 3


def test_model_codes():
    assert extract_model_codes("my dishwasher DW-6534 leaks") == ["DW-6534"]
    assert extract_model_codes("washer wm 8120 and fridge RF2290") == ["WM-8120", "RF-2290"]
    assert extract_model_codes("no code here") == []


def test_normalize():
    clean, info = normalize("My dishwahser DW-6534 is not draining!!!", return_info=True)
    assert clean == "my dishwasher is not draining"
    assert info["model_codes"] == ["DW-6534"]
    assert ("dishwahser", "dishwasher") in info["corrections"]
    assert normalize("what's the weather") == "what's the weather"      # real words are left alone
    assert normalize("pleeeease help") == "pleease help"


def test_small_talk():
    assert eliza_reply("hello") is not None
    assert eliza_reply("thanks a lot!") is not None
    assert eliza_reply("hi my oven is broken") is None                   # still goes to a team


def test_teams():
    assert team_for("dishwasher_issue") == "Appliance Service"
    assert team_for("card_declined") == "Credit Cards"
    assert team_for("oos") == HUMAN_AGENT


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
