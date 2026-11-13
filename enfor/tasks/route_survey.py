"""Route->Survey (Zhang, 2026): route descriptions with map questions (route-to-map, RS) and map descriptions with route
questions (map-to-route, SR), 4 area layouts x 400 test questions each, plus free-prose paraphrases of the route
descriptions (route-to-map only)."""

import json
import re

from enfor.env import DATA_DIR, DATASETS_DIR
from enfor.solvers import route

COLUMNS = {"route_rm": "RS", "route_mr": "SR", "route_rm_prose": "RS"}
DATA = DATASETS_DIR / "RouteSurvey" / "data"
SURVEY = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']

ROUTE = ['left', 'right', 'front', 'back']

DESC = {"R": "route", "S": "survey"}


def benchmark_items(combo, split="test"):
    """combo = <description><question>, e.g. RS = route description, survey question."""
    d, q = DESC[combo[0]], DESC[combo[1]]
    out = []
    for t in range(1, 5):
        for i, line in enumerate(open(DATA / f"template{t}_{q}_{split}.jsonl")):
            r = json.loads(line)
            out.append(dict(key=f"{combo}:{t}:{i}", combo=combo, template=t, description=r[d], question=r["question"],
                            answer=r["answer"], route_initial=r["route_initial"], up=r["up"]))
    return out


def items(column):
    X = benchmark_items(COLUMNS[column])
    if column != "route_rm_prose":
        return X
    prose = json.load(open(DATA_DIR / "route_survey_prose.json"))   # original description -> free-prose rewrite
    return [{**x, "description": prose[x["description"]], "key": "para:" + x["key"]} for x in X
            if x["description"] in prose]


def gold(column, x):
    return x["answer"]


# ---------------------------------------------------------------------------------------------- parser
_SYSTEM_BASE = """You transcribe a description of an area and a question into a JSON record for a route solver. Do not answer the question and do not convert directions: copy every direction exactly as the text states it (left / right / ahead / behind in a route description, compass words in a survey description).
Record: {"start": {"road": road you enter through or null, "facing": compass direction you face at the start if the text or the question states it, else null}, "events": [...], "question": {...}}
Events, in the order of the text:
- {"go": distance} for "Drive/Walk/Ride for d ..."; {"go": null} when you move on by an unstated distance (e.g. "continue ... and leave the town", "until you see ...").
- {"see": landmark, "side": "left"/"right" or a compass word} for a landmark you see there.
- {"turn": "left"/"right" or compass word, "onto": road} for every turn onto a road (including turns forced at a bend or a dead-end).
- {"back_to": road, "turn": direction} for "go back to where you leave <road> and turn <direction> back onto <road>"; {"end": road} for reaching the end of a road.
- {"hub": place, "roads": {road: direction}} when roads leave a central place (directions as stated: front / left / right / back relative to the road you face, or compass words); then {"onto": road} when you head out along one of them and {"return_to": place} when you come back.
- {"cross": road, "side": direction} for a road that meets the road you are on at this point and extends to that side; {"parallel": road, "side": direction} for a road running parallel far away on that side; {"boundary": road, "side": compass word} for a road or landmark stated to form that side of the area.
- {"close": road} when you arrive back at your starting point on that road, completing a tour.
Question:
- "In which direction is X from Y?" -> {"ask": "direction", "target": X, "from": Y}
- "If you go from A to B, where is X?" -> {"ask": "where", "target": X, "from": A, "to": B}
- "If C is on your left/right as you go along R, where is X?" -> {"ask": "where", "target": X, "along": R, "ref": C, "ref_pos": "left"/"right"}; with "what side of R is X on?" use "ask": "side".
- "If you go along R with C ahead / behind / on your left / on your right, what side of R is X on?" -> {"ask": "side", "target": X, "along": R, "ref": C, "ref_pos": "front"/"back"/"left"/"right"}
Use names exactly as written. Reply with only the JSON object."""

SHOTS = [
    ("""Description:
You enter Fern Town through Quarry Lane. Walk along Quarry Lane. Walk for 0.40 mile and you will see kiosk on your right. You continue walking along Quarry Lane and come to Mill Road on your left. Turn left onto Mill Road. Walk for 0.25 mile and you will see bakery on your left. Walk for 0.30 mile and you will see well on your right. You reach the end of Mill Road. Go back to where you leave Quarry Lane and turn left back onto Quarry Lane. Walk for 0.50 mile and you will see chapel on your left. You continue along Quarry Lane and leave the town.

Question:
If you go from bakery to well, where is Quarry Lane?""",
     {"start": {"road": "Quarry Lane", "facing": None},
      "events": [{"go": 0.40}, {"see": "kiosk", "side": "right"}, {"turn": "left", "onto": "Mill Road"},
                 {"go": 0.25}, {"see": "bakery", "side": "left"}, {"go": 0.30}, {"see": "well", "side": "right"},
                 {"end": "Mill Road"}, {"back_to": "Quarry Lane", "turn": "left"}, {"go": 0.50},
                 {"see": "chapel", "side": "left"}, {"go": None}],
      "question": {"ask": "where", "target": "Quarry Lane", "from": "bakery", "to": "well"}}),
    ("""Description:
You arrive at Pond Plaza. Extending from it are four roads: Ash Lane to the north, Elm Row to the south, Birch Way to the west, and Cliff Road to the east. From Pond Plaza, you head north along Ash Lane. Walk for 0.30 kilometre and you will see kiosk on the east. You then return along Ash Lane to Pond Plaza. From Pond Plaza, you head east along Cliff Road. Walk for 0.20 kilometre and you will see chapel on the north. You then return along Cliff Road to Pond Plaza. You finish your exploration at Pond Plaza area.

Question:
In which direction is chapel from kiosk?""",
     {"start": {"road": None, "facing": None},
      "events": [{"hub": "Pond Plaza", "roads": {"Ash Lane": "north", "Elm Row": "south", "Birch Way": "west",
                                                 "Cliff Road": "east"}},
                 {"onto": "Ash Lane"}, {"go": 0.30}, {"see": "kiosk", "side": "east"}, {"return_to": "Pond Plaza"},
                 {"onto": "Cliff Road"}, {"go": 0.20}, {"see": "chapel", "side": "north"}, {"return_to": "Pond Plaza"}],
      "question": {"ask": "direction", "target": "chapel", "from": "kiosk"}}),
    ("""Description:
You enter Reed Town through Stone Road. At this point, you see Lake Shore on your right, intersecting with Stone Road. Drive along Stone Road. Drive for 0.50 kilometre and you will see inn on your right. You continue driving on Stone Road and come to, on your right, Hill Lane. Turning right onto Hill Lane. Drive for 1.00 kilometre and you will see barn on your left. Continue driving on Hill Lane until you are forced to make a left turn. Turning left onto Glen Road. Drive for 0.70 kilometre and you will see mill on your right. Continue driving along Glen Road until you are again forced to make a left turn. Turning left onto Vale Road. Continue driving along Vale Road until you meet a dead-end into Stone Road. Turn right onto Stone Road. You drive along Stone Road until you see Pine Ridge on your right, intersecting with the one you are on. Oak Forest runs parallel to your far right, on the other side of the town.

Question:
Assume that at the start of the description you are facing south. Use this information to answer the following question. In which direction is barn from Oak Forest?""",
     {"start": {"road": "Stone Road", "facing": "south"},
      "events": [{"cross": "Lake Shore", "side": "right"}, {"go": 0.50}, {"see": "inn", "side": "right"},
                 {"turn": "right", "onto": "Hill Lane"}, {"go": 1.00}, {"see": "barn", "side": "left"},
                 {"turn": "left", "onto": "Glen Road"}, {"go": 0.70}, {"see": "mill", "side": "right"},
                 {"turn": "left", "onto": "Vale Road"}, {"turn": "right", "onto": "Stone Road"}, {"go": None},
                 {"cross": "Pine Ridge", "side": "right"}, {"parallel": "Oak Forest", "side": "right"}],
      "question": {"ask": "direction", "target": "barn", "from": "Oak Forest"}}),
]

PARSER_SYSTEM = _SYSTEM_BASE.replace(
    '- {"back_to": road, "turn": direction} for "go back to where you leave <road> and turn <direction> back onto <road>";',
    '- {"back_to": road, "turn": direction} for "go back to where you leave <road> and turn <direction> back onto <road>" '
    '(also when the direction is a compass word: "turn east back onto Oak Road" -> {"back_to": "Oak Road", "turn": "east"});'
).replace(
    '- {"close": road} when you arrive back at your starting point on that road, completing a tour.',
    '- {"close": road} when you arrive back at your starting point on that road, completing a tour (also "continue until '
    'you complete the tour and return to your starting point on <road>").\nWrite every sentence that names a side of the '
    'area ("The east is made up of X") as a boundary event, and fill start.facing from "Drive/Walk <compass> along <road>".')
SURVEY_SHOT = ("""Description:
Brook Town is surrounded by four major landmarks: Kiln Road, Moss Lane, Bay Shore and Elm Ridge. The east is made up of the Kiln Road, running north-south. The south is made up of the Moss Lane, running west-east. The north is made up of the Bay Shore, running west-east. The west is made up of the Elm Ridge, running north-south. You enter the town through Kiln Road. Walk north along Kiln Road. Walk for 0.20 kilometre and you will see inn on the west. You continue walking on Kiln Road and come to, on the west, Fen Lane. Turning west onto Fen Lane. Walk for 0.60 kilometre and you will see barn on the north. Continue walking on Fen Lane until you are forced to make a turn to the north. Turning north onto Ash Row. Walk for 0.40 kilometre and you will see mill on the east. Continue walking along Ash Row until you are again forced to make a turn to the east. Turning east onto Cove Lane. Continue walking along Cove Lane until you meet a dead-end into Kiln Road. Turn north onto Kiln Road. Continue walking north along Kiln Road and you leave the town.

Question:
If barn is on your left as you go along Fen Lane, where is Kiln Road?""",
           {"start": {"road": "Kiln Road", "facing": "north"},
            "events": [{"boundary": "Kiln Road", "side": "east"}, {"boundary": "Moss Lane", "side": "south"},
                       {"boundary": "Bay Shore", "side": "north"}, {"boundary": "Elm Ridge", "side": "west"},
                       {"go": 0.20}, {"see": "inn", "side": "west"}, {"turn": "west", "onto": "Fen Lane"}, {"go": 0.60},
                       {"see": "barn", "side": "north"}, {"turn": "north", "onto": "Ash Row"}, {"go": 0.40},
                       {"see": "mill", "side": "east"}, {"turn": "east", "onto": "Cove Lane"},
                       {"turn": "north", "onto": "Kiln Road"}, {"go": None}],
            "question": {"ask": "where", "target": "Kiln Road", "along": "Fen Lane", "ref": "barn", "ref_pos": "left"}})


def _prefix():
    out = []
    for u, r in SHOTS + [SURVEY_SHOT]:
        out += [{"role": "user", "content": u}, {"role": "assistant", "content": json.dumps(r)}]
    return out


def _user(x):
    extra = (f'Assume that at the start of the description you are facing {x["route_initial"]}. Use this information to '
             f'answer the following question. ') if x["combo"] == "RS" else ""
    return f"Description:\n{x['description']}\n\nQuestion:\n{extra}{x['question']}"


def parser_prompt(column, X):
    return dict(system=PARSER_SYSTEM, prefix=_prefix(), requests=[(x["key"], _user(x)) for x in X], effort="low",
                max_tokens=8192)


def _cot_user(x, cot=False):
    extra = (f'Assume that at the start of the description you are facing {x["route_initial"]}. Use this information to '
             f'answer the following question. ') if x["combo"] == "RS" else ""
    if x["combo"][1] == "R":
        choices = 'Choose answer from left, right, front and back.'
    else:
        choices = 'Choose answer from north, south, east, west, northeast, northwest, southeast, southwest.'
    tail = (" Let's think step by step by tracking the location and direction changes described in the description, then "
            "end with 'Answer: <your answer>'.") if cot else " Only give your answer, as 'Answer: <your answer>'."
    return (f"You will read a description of an area and answer a question.\n\nDescription:\n{x['description']}\n\n"
            f"Question:\n{extra}{x['question']}\n{choices}{tail}")


def cot_prompt(column, X):
    return dict(system="", prefix=[], requests=[(x["key"], _cot_user(x, True)) for x in X], effort="low",
                max_tokens=8192)


def read_parse(text):
    text = text or ""
    if "RECORD" in text:
        text = text[text.rfind("RECORD"):]
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


def decide(column, text, x):
    try:
        return route.answer(read_parse(text))
    except Exception:
        return None


def read_cot(column, text, x):
    labels = ROUTE if x["combo"][1] == "R" else SURVEY
    t = (text or "").lower().replace("north-east", "northeast").replace("north-west", "northwest") \
        .replace("south-east", "southeast").replace("south-west", "southwest").replace("behind", "back")
    m = re.findall(r"answer\s*[:\uff1a]?\s*\**\s*([a-z]+)", t)
    if m and m[-1] in labels:
        return m[-1]
    hits = [w for w in re.findall(r"[a-z]+", t) if w in labels]
    return hits[-1] if hits else None
