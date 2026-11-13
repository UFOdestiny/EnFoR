"""SpaceNLI (Abzianidze et al., 2023): spatial natural language inference over 160 templates; 1,000 problems sampled
from the released split (seed 1)."""

import json
import random

from enfor.env import DATASETS_DIR
from enfor.solvers import spacenli
from enfor.tasks.common import COT_SYSTEM, parse_record, read_cot as _read_cot, user_text

COLUMNS = {"spacenli": "spacenli"}
COL = "spacenli"


def items(column):
    from datasets import load_from_disk

    d = load_from_disk(str(DATASETS_DIR / "SpaceNLI" / "hf_dataset"))["train"]
    idx = sorted(random.Random(1).sample(range(len(d)), 1000))
    return [dict(key=f"nlit{i}", context=d[i]["premises"].strip(), question=d[i]["hypothesis"].strip(), options=None,
                 gold=d[i]["label"], meta={"id": d[i]["id"], "template": d[i]["id"].rsplit("-", 1)[0]}) for i in idx]


def gold(column, x):
    return x["gold"]


# ---------------------------------------------------------------------------------------------- parser
PARSER_SYSTEM = """Transcribe the premise and the hypothesis into logical atoms for a solver; do not decide the answer and do not draw inferences. Write only what each sentence itself states, and keep every direction in the frame it is stated in.
Reply with only a JSON object {"named_places": [...], "premises": [atom, ...], "hypothesis": [atom, ...]}.
- named_places: proper names of places (cities, countries, regions) that occur.
- atom: {"rel": R, "args": [...], "neg": true if the sentence negates it, "time": "now" | "past" | "before", "aspect": "progressive" | "completed" (motion only)}.
  time: "now" for present tense (is, is walking, has walked), "past" for simple past (was, walked), "before" for a state before the event.
- R for states (args [A, B] unless noted; A is located relative to B):
  in (in, inside, within), outside, at, on, on_top_of, under, above, below,
  in_front_of / behind (args [A, B, V]: V is the viewer if the sentence names one, else null; without a viewer they are relative to B's own front),
  left_of / right_of (args [A, B, V]: seen from viewer V, null if not named),
  between (args [A, B, C]: A is between B and C), near (near, close to), far_from, with, different (A and B are different places or persons).
- R for motion of a mover A relative to a place B (args [A, B]): into (into, enter), out_of (out of, leave), from, to, through, across, towards, away_from, by (by, past).
  Write one motion atom per moving participant the sentence states (who moves is part of reading the verb: whoever is carried, thrown, escorted or sent also moves).
- A locative phrase gives an atom only for the participant the sentence places there.
- Anything else (counts like twice, wanting, seeing, mentioning, visibility, opposite sides, parts) is {"rel": "other", "text": the words}.
Names as written, without "the"."""

# (user text, parse, parse in one absolute frame for the LLM-converts baseline)
SHOTS = [
    ("Premise: Linden is in Ostland. Kade is not in Ostland. The courier is cycling from Linden to Kade.\nHypothesis: The courier is cycling from Ostland to Kade.",
     {"named_places": ["Linden", "Ostland", "Kade"],
      "premises": [{"rel": "in", "args": ["Linden", "Ostland"], "neg": False, "time": "now"},
                   {"rel": "in", "args": ["Kade", "Ostland"], "neg": True, "time": "now"},
                   {"rel": "from", "args": ["courier", "Linden"], "neg": False, "time": "now", "aspect": "progressive"},
                   {"rel": "to", "args": ["courier", "Kade"], "neg": False, "time": "now", "aspect": "progressive"}],
      "hypothesis": [{"rel": "from", "args": ["courier", "Ostland"], "neg": False, "time": "now", "aspect": "progressive"},
                     {"rel": "to", "args": ["courier", "Kade"], "neg": False, "time": "now", "aspect": "progressive"}]},
     {"named_places": ["Linden", "Ostland", "Kade"],
      "premises": [{"rel": "in", "args": ["Linden", "Ostland"], "neg": False, "time": "now"},
                   {"rel": "in", "args": ["Kade", "Ostland"], "neg": True, "time": "now"},
                   {"rel": "from", "args": ["courier", "Linden"], "neg": False, "time": "now", "aspect": "progressive"},
                   {"rel": "to", "args": ["courier", "Kade"], "neg": False, "time": "now", "aspect": "progressive"}],
      "hypothesis": [{"rel": "from", "args": ["courier", "Ostland"], "neg": False, "time": "now", "aspect": "progressive"},
                     {"rel": "to", "args": ["courier", "Kade"], "neg": False, "time": "now", "aspect": "progressive"}]}),
    ("Premise: The lamp is in front of the sofa. The rug is between the lamp and the sofa.\nHypothesis: The rug is in front of the sofa.",
     {"named_places": [],
      "premises": [{"rel": "in_front_of", "args": ["lamp", "sofa", None], "neg": False, "time": "now"},
                   {"rel": "between", "args": ["rug", "lamp", "sofa"], "neg": False, "time": "now"}],
      "hypothesis": [{"rel": "in_front_of", "args": ["rug", "sofa", None], "neg": False, "time": "now"}]},
     {"named_places": [],
      "premises": [{"rel": "north_of", "args": ["lamp", "sofa"], "neg": False, "time": "now"},
                   {"rel": "between", "args": ["rug", "lamp", "sofa"], "neg": False, "time": "now"}],
      "hypothesis": [{"rel": "north_of", "args": ["rug", "sofa"], "neg": False, "time": "now"}]}),
    ("Premise: Pia has carried the lamp out of the attic.\nHypothesis: The lamp is in the attic.",
     {"named_places": [],
      "premises": [{"rel": "out_of", "args": ["Pia", "attic"], "neg": False, "time": "now", "aspect": "completed"},
                   {"rel": "out_of", "args": ["lamp", "attic"], "neg": False, "time": "now", "aspect": "completed"}],
      "hypothesis": [{"rel": "in", "args": ["lamp", "attic"], "neg": False, "time": "now"}]},
     None),
]


def parser_prompt(column, X):
    prefix = []
    for u, rec, _ in SHOTS:
        prefix += [{"role": "user", "content": u}, {"role": "assistant", "content": json.dumps(rec)}]
    return dict(system=PARSER_SYSTEM, prefix=prefix, effort="low", max_tokens=8192,
                requests=[(x["key"], f"Premise: {x['context']}\nHypothesis: {x['question']}") for x in X])


def cot_prompt(column, X):
    return dict(system=COT_SYSTEM, prefix=[], requests=[(x["key"], user_text(COL, x)) for x in X], effort="low",
                max_tokens=8192)


def decide(column, text, x):
    try:
        return spacenli.decide(parse_record(text))
    except Exception:
        return None


def read_cot(column, text, x):
    return _read_cot(COL, text, x)
