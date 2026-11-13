"""Answer formats and readers shared by the multiple-choice and NLI columns (SpatialEval maze, SpaceNLI)."""

import json
import re

SPEC = {'spatialeval_maze': 'Answer with the option letter.', 'spacenli': 'Answer with one of: entailment, neutral, contradiction.'}
COT_SYSTEM = "Solve the spatial reasoning problem. Think step by step, then end with a final line 'Answer: <answer>'."


def options_text(o):
    return "\n".join(f"{k}. {v}" for k, v in o.items())


def user_text(col, x):
    """The item as every method reads it."""
    if col == "spatialeval_maze":
        return f"{x['context']}\nAvailable options:\n{options_text(x['options'])}\n{SPEC[col]}"
    return f"Premise: {x['context']}\nHypothesis: {x['question']}\n{SPEC[col]}"


def norm_answer(col, s, x=None):
    """Normalise a raw answer string to the column's gold format, or None."""
    if s is None:
        return None
    s = str(s).strip().strip("*").strip().lower().replace("‑", "-").replace("_", "-")
    s = s.strip(" .'\"`")
    if col == "stepgame":
        s = s.replace("upper left", "upper-left").replace("upper right", "upper-right") \
             .replace("lower left", "lower-left").replace("lower right", "lower-right")
        m = re.search(r"\b(upper-left|upper-right|lower-left|lower-right|above|below|left|right|overlap)\b", s)
        return m.group(1) if m else None
    if col == "babi17":
        m = re.search(r"\b(yes|no)\b", s)
        return m.group(1) if m else None
    if col == "babi19":
        ws = re.findall(r"\b(north|south|east|west|n|s|e|w)\b", s.replace(",", " "))
        full = {"n": "north", "s": "south", "e": "east", "w": "west"}
        ws = [full.get(w, w) for w in ws]
        return " ".join(ws) if ws else None
    if col == "spacenli":
        m = re.search(r"\b(entailment|neutral|contradiction|entails|contradicts)\b", s)
        return None if not m else {"entails": "entailment", "contradicts": "contradiction"}.get(m.group(1), m.group(1))
    # option-letter columns: a letter, or an option text
    m = re.match(r"^\(?([a-d])\)?(\b|$)", s)
    if m:
        return m.group(1).upper()
    if x is not None:
        for k, v in x["options"].items():
            if s == str(v).strip().lower():
                return k
    return None


def read_cot(col, t, x=None):
    """Last 'Answer: ...' line of a reply; else 'Answer <x>' / 'the answer is <x>'; else a reply that is only the
    answer (gpt-oss sometimes ends with the bare answer)."""
    t = (t or "").replace("**", "")
    for pat in (r"answer\s*[:：]\s*(.+)", r"\banswer(?:\s+is)?\s+(.+)"):
        m = re.findall(pat, t, re.I)
        if m:
            a = norm_answer(col, m[-1], x)
            if a is not None:
                return a
    if len(t.strip()) <= 40:
        return norm_answer(col, t, x)
    return None


def parse_record(t):
    t = t or ""
    i = t.rfind("RECORD:")
    t = t[i + 7:] if i >= 0 else t
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
