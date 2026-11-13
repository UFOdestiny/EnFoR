"""RoomSpace (Li et al., 2024), side view: the reader stands at a wall's door looking inwards and relations are stated
from that view; yes/no questions ask whether a layout is possible. 1,000 test questions stratified by setting
(seed 1)."""

import json
import random
import re

from enfor.env import DATASETS_DIR
from enfor.solvers import room_grid

COLUMNS = {"roomspace": "sd"}
DIRS = ["north", "south", "east", "west", "north-east", "north-west", "south-east", "south-west"]


def items(column, seed=1, n=1000):
    from datasets import load_from_disk

    ds = load_from_disk(str(DATASETS_DIR / "RoomSpace" / "hf_dataset"))["test"]
    idx = [i for i, r in enumerate(ds) if r["view"] == "sd" and r["q_type"] == "YN" and r["answer"]]
    by = {}
    for i in idx:
        by.setdefault(ds[i]["setting"], []).append(i)
    rng = random.Random(seed)
    per = n // len(by)
    pick = []
    for s in sorted(by):
        pick += rng.sample(by[s], per)
    return [dict(key=f"rs:{i}", story=ds[i]["story"], question=ds[i]["question"], answer=ds[i]["answer"][0],
                 setting=ds[i]["setting"]) for i in pick]


def gold(column, x):
    return x["answer"]


# ---------------------------------------------------------------------------------------------- parser
EX_STORY = ("Imagine a square-shaped study, bordered by four walls. This room contains a collection of furniture, including "
            "a desk placed in the north-west, a lamp, and a bookshelf placed in the east. Imagine yourself at the southern "
            "wall's door, looking inwards. From this perspective, the lamp is in front of and to the right of the desk, at a "
            "short distance. The bookshelf is behind the lamp.")

EX_Q = "From this perspective, could the desk be positioned to the left of of the bookshelf?"

SPEC_COMMON = ("Reply with only a JSON object: {\"regions\": [[object, region]], \"walls\": [object], \"relations\": "
               "[[A, REL, B, distance or null]], \"question\": [A, REL, B]}. regions: the room region an object is placed in "
               "(north, north-east, ..., center), as written. walls: objects stated to be against the wall. distance: short, "
               "moderate or far if stated.")

PARSER_SYSTEM = ("You transcribe a room description. Do not convert perspectives: copy each relation with the words used. "
              "REL is the list of perspective words used, from {front, behind, left, right} (e.g. 'in front of and to the "
              "left of' -> [\"front\", \"left\"]). " + SPEC_COMMON.replace("{\"regions\"", "{\"viewer\": the wall whose "
              "door the reader stands at (south, north, east or west), \"regions\""))

PARSER_SHOT = {"viewer": "south", "regions": [["desk", "north-west"], ["bookshelf", "east"]], "walls": [],
            "relations": [["lamp", ["front", "right"], "desk", "short"], ["bookshelf", ["behind"], "lamp", None]],
            "question": ["desk", ["left"], "bookshelf"]}

COT_SYSTEM = ("Answer the question about the room. Think step by step, then end with 'Answer: Yes' or 'Answer: No'.")


def _user(x):
    return f"{x['story']}\nQuestion: {x['question']}"


def parser_prompt(column, X):
    prefix = [{"role": "user", "content": f"{EX_STORY}\nQuestion: {EX_Q}"},
              {"role": "assistant", "content": json.dumps(PARSER_SHOT)}]
    return dict(system=PARSER_SYSTEM, prefix=prefix, requests=[(x["key"], _user(x)) for x in X], effort="low",
                max_tokens=4096)


def cot_prompt(column, X):
    return dict(system=COT_SYSTEM, prefix=[], requests=[(x["key"], _user(x)) for x in X], effort="low",
                max_tokens=6144)


def to_text(rec, kind):
    """Record -> top-down text and question for the exact solver (None if unusable)."""
    facing = {"south": "north", "north": "south", "west": "east", "east": "west"}.get(str(rec.get("viewer", "")).lower())

    def rel(r):
        if kind == "absolute":
            r = str(r).lower().replace(" ", "-")
            return r if r in DIRS else None
        if facing is None:
            return None
        words = {str(w).lower() for w in (r if isinstance(r, list) else [r])} & {"front", "behind", "left", "right"}
        return room_grid.ego_to_compass(words, facing) if words else None
    objs = []
    for o, _ in rec.get("regions") or []:
        objs.append(o)
    for a, _, b, *_ in rec.get("relations") or []:
        objs += [a, b]
    seen = []
    for o in objs:
        if o not in seen:
            seen.append(o)
    reg = {o: g for o, g in rec.get("regions") or []}
    items = ", ".join(f"a {o}" + (f" placed in the {reg[o]}" if o in reg else "") for o in seen)
    sents = [f"Imagine a square-shaped room, bordered by four walls. This room contains a collection of furniture, including {items}."]
    for a, r, b, *d in rec.get("relations") or []:
        c = rel(r)
        if c is None:
            return None, None
        dist = d[0] if d and d[0] else None
        sents.append(f"The {a} is placed to the {c} of the {b}" + (f", at a {dist} distance." if dist else "."))
    a, r, b = rec["question"]
    c = rel(r)
    if c is None:
        return None, None
    return " ".join(sents), f"Could the {a} be placed to the {c} of the {b}?"


_SOLVER = []


def decide(column, text, x):
    if not _SOLVER:
        _SOLVER.append(room_grid.RoomSpaceFormalizer())
    F = _SOLVER[0]
    try:
        rec = json.loads(re.search(r"\{.*\}", text or "", re.S).group(0))
        story, q = to_text(rec, "delegate")
        if q:
            st = F.choice_status(F.scene(story, q), q, ["Yes", "No"], False)
            return "Yes" if st["Yes"] == room_grid.ENTAILED else "No" if st["No"] == room_grid.ENTAILED else None
    except Exception:
        return None
    return None


def read_cot(column, text, x):
    m = re.findall(r"answer\s*[:\uff1a]?\s*\**\s*(yes|no)", (text or "").lower())
    return m[-1].capitalize() if m else None
