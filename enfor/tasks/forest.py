"""FoREST (Premsri and Kordjamshidi, 2025): one relation between two objects, stated relative to the camera, to the
relatum, or with the frame left open; the question asks for the relation from the camera's or the relatum's
perspective. Columns: the ambiguous A-split (F1 of "several possible answers") and the C-split with a stated frame
(accuracy). Sample: 1,000 C-split camera + 1,000 C-split relatum + 500 + 500 A-split items (seed 0)."""

import ast
import json
import random
import re

from enfor.env import DATASETS_DIR
from enfor.solvers import frames

COLUMNS = {"forest_amb_f1": "A", "forest_stated": "C"}
DATA = DATASETS_DIR / "FoREST"
SPLITS = {"C_camera": ("C-split_QA_camera_total", 1000), "C_relatum": ("C-split_QA_relatum_total", 1000),
          "A_camera": ("A-split_QA_camera_total", 500), "A_relatum": ("A-split_QA_relatum_total", 500)}


def forest_prompts():
    """FoREST's prompt module (utils/prompt.py), read as data without importing or executing it."""
    out = {}
    for node in ast.parse((DATA / "utils" / "prompt.py").read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def sample():
    rng = random.Random(0)
    out = []
    for tag, (f, n) in SPLITS.items():
        data = json.load(open(DATA / "Dataset" / f"{f}.json"))["data"]
        for e in rng.sample(data, n):
            out.append({**e, "split": tag, "key": f"{tag}:{e['id']}"})
    return out


def items(column):
    return [e for e in sample() if e["split"].startswith(COLUMNS[column])]


def gold(column, x):
    return tuple(x["candidate_answer"]) if column == "forest_amb_f1" else x["candidate_answer"][0]


# ---------------------------------------------------------------------------------------------- parser
# The parser first copies the words of the relation sentence that state whose perspective it is from, then reads the
# frame off them; without such words the frame is unspecified and the solver returns both readings.
_SYSTEM_BASE = (
    "You transcribe one spatial scene description into a record; you do not answer the question and you do not convert "
    "between perspectives. Copy what the text says and do not fill in what it leaves open.\n"
    "Fields, in this order:\n"
    "- relation: the direction word the context uses for the first object relative to the second: left, right, front "
    "or back (as written, even if 'inside and at the right of').\n"
    "- perspective: copy the words of the sentence giving that relation which say whose perspective it is from, e.g. "
    "'relative to the camera', 'relative to observer', 'from my perspective', 'from my point of view', 'relative to the "
    "car', 'from the car's perspective', 'at the car's own left side'; or '' if that sentence has no such words. Only "
    "that sentence counts: a sentence saying which way an object faces, and the question, never count.\n"
    "- frame: read off perspective only: 'camera' if it names the camera, observer, viewer, me or my; 'relatum' if it "
    "names the second object; 'unspecified' if perspective is ''. Never choose a frame from the direction word, the "
    "kind of object or the question.\n"
    "- facing: which way the second object faces, as written relative to the camera / observer: 'toward' (facing toward "
    "the camera), 'away', 'left', 'right', or 'none' if the text does not say.\n"
    "- view: whose perspective the question asks from: 'camera' (the camera, observer, viewer or me) or 'relatum' (the "
    "second object's perspective, view or angle).\n"
    'Reply with only a JSON object, e.g. {"relation": "left", "perspective": "", "frame": "unspecified", '
    '"facing": "right", "view": "camera"}.')

PARSER_SYSTEM = _SYSTEM_BASE.replace(
    "'camera' if it names the camera, observer, viewer, me or my;",
    "'camera' if it names the camera, observer, viewer, me or my, with or without 'the' (e.g. 'relative to camera');"
).replace(
    "second object's perspective, view or angle).",
    "second object's perspective, view or angle). Words such as 'from the second object's position' only say where to "
    "measure from, not whose perspective.")

# perspective words of FoREST's four few-shot contexts (copied from their relation sentences)
FOREST_PHRASES = ["relative to the car", "from the car's perspective", "from the observer's perspective",
                  "from my perspective"]

# records for FoREST's four few-shot contexts
FOREST_RECORDS = [{"relation": "front", "frame": "relatum", "facing": "toward", "view": "camera"},
                  {"relation": "left", "frame": "relatum", "facing": "right", "view": "camera"},
                  {"relation": "right", "frame": "camera", "facing": "none", "view": "camera"},
                  {"relation": "left", "frame": "camera", "facing": "right", "view": "camera"}]
# synthetic contexts written for this prompt (no FoREST item text), unspecified frame
SYNTH_SHOTS = [
    ("Context: A lamp is to the right of a sofa. The sofa is facing away from the camera. Question: From the camera's "
     "perspective, where is the lamp relative to the sofa?",
     {"relation": "right", "perspective": "", "frame": "unspecified", "facing": "away", "view": "camera"}),
    ("Context: A kettle is in front of a stove. The stove is facing left relative to the camera. Question: From the "
     "stove's perspective, where is the kettle relative to the stove?",
     {"relation": "front", "perspective": "", "frame": "unspecified", "facing": "left", "view": "relatum"}),
    ("Context: A vase is behind a barrel. Question: From the observer's perspective, what is the spatial relation of "
     "the vase to the barrel?",
     {"relation": "back", "perspective": "", "frame": "unspecified", "facing": "none", "view": "camera"}),
]


def _msg(u, r):
    return [{"role": "user", "content": u}, {"role": "assistant", "content": json.dumps(r)}]


def parser_shots():
    users = [m["content"] for m in forest_prompts()["QA_few_shot"] if m["role"] == "user"]
    forest = [(u, {"relation": r["relation"], "perspective": ph, **{k: r[k] for k in ("frame", "facing", "view")}})
              for u, r, ph in zip(users, FOREST_RECORDS, FOREST_PHRASES)]
    order = [forest[0], SYNTH_SHOTS[0], forest[1], SYNTH_SHOTS[1], forest[2], SYNTH_SHOTS[2], forest[3]]
    return [m for u, r in order for m in _msg(u, r)]


def _user(e):
    return f"Context: {e['context']} Question: {e['question']}"


def parser_prompt(column, X):
    return dict(system=PARSER_SYSTEM, prefix=parser_shots(), requests=[(e["key"], _user(e)) for e in X],
                effort="low", max_tokens=2048)


def cot_prompt(column, X):
    P = forest_prompts()
    return dict(system=P["QA_prompt_COT"], prefix=P["QA_COT_ex"], requests=[(e["key"], _user(e)) for e in X],
                effort="low", max_tokens=4096)


def norm_relation(w):
    """Map the direction phrase as written to one of the four labels (solver-side lexicon)."""
    w = w.lower()
    for key, words in (("back", ("behind", "back", "rear")), ("front", ("front", "ahead")),
                       ("left", ("left",)), ("right", ("right",))):
        if any(x in w for x in words):
            return key
    return w


def possible_answers(text):
    """All answers the parse allows (two readings when the frame is unspecified and the facing is known)."""
    try:
        r = json.loads(re.search(r"\{.*\}", text or "", re.S).group(0))
        rel, frame, facing, view = (str(r[k]).lower() for k in ("relation", "frame", "facing", "view"))
        rel = norm_relation(rel)
        facing = None if facing in ("none", "null", "") else facing
        if view == "relatum" and facing is None:
            return []
        readings = {"camera": ["relative"], "relatum": ["intrinsic"]}.get(frame)
        if readings is None:   # unspecified: the intrinsic reading needs a facing
            readings = ["intrinsic", "relative"] if facing else ["relative"]
        return [frames.solve(rel, fr, facing, view) for fr in readings]
    except Exception:
        return []


def decide(column, text, x):
    s = sorted(set(possible_answers(text)))
    if column == "forest_stated":
        return s[0] if len(s) == 1 else None
    return tuple(s) or None


def read_cot(column, text, x):
    m = re.findall(r"answer\s*[:\-]?\s*\**\s*(left|right|front|back)", (text or "").lower())
    v = m[-1] if m else None
    if v is None:
        m = re.findall(r"\b(left|right|front|back)\b", (text or "").lower())
        v = m[-1] if m else None
    if column == "forest_stated":
        return v
    return (v,) if v else None


def f1(gm, pm):
    tp = sum(g and p for g, p in zip(gm, pm))
    fp = sum(p and not g for g, p in zip(gm, pm))
    fn = sum(g and not p for g, p in zip(gm, pm))
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    return 2 * pr * rc / max(1e-9, pr + rc)


def metric(column, X, pred):
    if column == "forest_amb_f1":   # F1 of "several possible answers"
        gm = [len(set(x["candidate_answer"])) > 1 for x in X]
        pm = [bool(pred.get(x["key"])) and len(set(pred[x["key"]])) > 1 for x in X]
        return f1(gm, pm)
    return sum(pred.get(x["key"]) == gold(column, x) for x in X) / len(X)
