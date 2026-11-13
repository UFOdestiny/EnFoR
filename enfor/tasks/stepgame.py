"""StepGame answerability: StepGame stories restated in the agents' intrinsic frames with some facing sentences removed
(held-out set, two sentence-aligned paraphrases, free prose). Gold labels come from the generator's facts; a question
is answerable iff every consistent completion of the unstated facings gives the same relation."""

import json
import re

from enfor.env import DATA_DIR
from enfor.solvers import facings

COLUMNS = {"sg_heldout": "heldout", "sg_para1": "paraphrase1", "sg_para2": "paraphrase2", "sg_prose": "prose"}
UNKNOWN = facings.UNKNOWN
ANSWERS = set(facings.LABELS) | {UNKNOWN}

COT_SYSTEM = ("Answer the question about the agents. Each agent faces a compass direction if stated; positions are "
              "given from that agent's own point of view. If the description does not determine the answer, say so. "
              "Think step by step, then end with 'Answer: <relation>' (one of above, below, left, right, upper-left, "
              "upper-right, lower-left, lower-right, overlap; above is north, left is west) or 'Answer: cannot be "
              "determined'.")

PARSER_SYSTEM = ("Transcribe the description; do not answer and do not convert directions. Reply with only a JSON object: "
              "{\"facings\": {agent: compass direction as stated, or null if the text does not state it}, "
              "\"statements\": [[A, [words], B]] where words are the position words relative to B's own view, from "
              "front, behind, left, right, \"question\": [A, B]} for 'What is the relation of the agent A to the agent B?'. "
              "List every agent mentioned in facings.")

PARSER_SHOT = ("Every agent faces the compass direction stated for it, and a position given relative to an agent is seen "
            "from that agent's own point of view. Answers use compass directions: above is north and left is west. Q "
            "faces east. R is in front of Q and on its left. T is behind S. V is on U's right. U is facing west.\n"
            "What is the relation of the agent V to the agent Q?",
            {"facings": {"Q": "east", "R": None, "S": None, "T": None, "U": "west", "V": None},
             "statements": [["R", ["front", "left"], "Q"], ["T", ["behind"], "S"], ["V", ["right"], "U"]],
             "question": ["V", "Q"]})


def items(column):
    return json.load(open(DATA_DIR / "stepgame_answerability" / f"{COLUMNS[column]}.json"))


def gold(column, x):
    return x["gold"]


def _user(x):
    return f"{x['context']}\n{x['question']}"


def parser_prompt(column, X):
    prefix = [{"role": "user", "content": PARSER_SHOT[0]}, {"role": "assistant", "content": json.dumps(PARSER_SHOT[1])}]
    return dict(system=PARSER_SYSTEM, prefix=prefix, requests=[(x["key"], _user(x)) for x in X], effort="low",
                max_tokens=4096)


def cot_prompt(column, X):
    return dict(system=COT_SYSTEM, prefix=[], requests=[(x["key"], _user(x)) for x in X], effort="low",
                max_tokens=8192)


def decide(column, text, x):
    v = facings.decide(text)
    return v if v in ANSWERS else None


def read_cot(column, text, x):
    t = (text or "").lower().replace("\u2011", "-")
    m = re.findall(r"answer:\s*\**\s*(cannot be determined|undetermined|upper-left|upper-right|lower-left|lower-right|"
                   r"above|below|left|right|overlap)", t)
    if not m:
        return None
    return UNKNOWN if m[-1] in ("cannot be determined", "undetermined") else m[-1]
