"""Exact entailment for relations stated in agents' intrinsic frames with possibly unstated facings (StepGame
answerability).

A parse lists each agent's facing (a compass word, or null when the text does not state it) and statements
[A, words, B]: A is at <words> (front / behind / left / right) from B's own point of view. Facings are variables over
the four compass directions, positions are integers, and each statement constrains A - B to B's rotated word vector.
A relation is the answer if the constraints entail it (its negation is unsatisfiable); if none is entailed, the
question is unanswerable.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

UNKNOWN = "cannot be determined"
LABELS = ["above", "below", "left", "right", "upper-left", "upper-right", "lower-left", "lower-right", "overlap"]
SIGN_LABEL = {(0, 1): "above", (0, -1): "below", (-1, 0): "left", (1, 0): "right", (-1, 1): "upper-left",
              (1, 1): "upper-right", (-1, -1): "lower-left", (1, -1): "lower-right", (0, 0): "overlap"}
FACE = {"north": (0, 1), "east": (1, 0), "south": (0, -1), "west": (-1, 0)}
FACE_INDEX = {"north": 0, "east": 1, "south": 2, "west": 3}

Statement = Tuple[str, str, int, int]   # (locatum, relatum, ahead, right) with ahead, right in {-1, 0, 1}


def entailed_relation(facing: Dict[str, str], statements: List[Statement], a: str, b: str) -> Optional[str]:
    """The relation of a to b entailed by every consistent completion, UNKNOWN if none is, None if inconsistent."""
    import z3

    names = set(facing) | {s for s, _, _, _ in statements} | {o for _, o, _, _ in statements} | {a, b}
    x = {n: z3.Int("x_" + n) for n in names}
    y = {n: z3.Int("y_" + n) for n in names}
    f = {n: z3.Int("f_" + n) for n in names}
    s = z3.Solver()
    for n in names:
        s.add(f[n] >= 0, f[n] <= 3)
        if facing.get(n) in FACE_INDEX:
            s.add(f[n] == FACE_INDEX[facing[n]])
    for loc, rel, ah, rt in statements:
        fx = z3.If(f[rel] == 1, 1, z3.If(f[rel] == 3, -1, 0))
        fy = z3.If(f[rel] == 0, 1, z3.If(f[rel] == 2, -1, 0))
        s.add(x[loc] == x[rel] + ah * fx + rt * fy, y[loc] == y[rel] + ah * fy - rt * fx)
    if s.check() != z3.sat:
        return None
    dx, dy = x[a] - x[b], y[a] - y[b]
    sg = lambda v: z3.If(v > 0, 1, z3.If(v < 0, -1, 0))  # noqa: E731
    # agents that are not connected are unconstrained, so no relation is entailed
    for (cx, cy), name in SIGN_LABEL.items():
        s.push()
        s.add(z3.Not(z3.And(sg(dx) == cx, sg(dy) == cy)))
        if s.check() == z3.unsat:
            s.pop()
            return name
        s.pop()
    return UNKNOWN


def _words(words) -> Tuple[int, int]:
    ws = " ".join(str(w).lower() for w in (words if isinstance(words, list) else [words]))
    ahead = 1 if ("front" in ws or "ahead" in ws) else -1 if ("behind" in ws or "back" in ws) else 0
    right = 1 if "right" in ws else -1 if "left" in ws else 0
    return ahead, right


def read_parse(text: str):
    """(facing, statements, a, b) from the parser's JSON reply."""
    r = json.loads(re.search(r"\{.*\}", text or "", re.S).group(0))
    facing = {str(k).lower(): str(v).lower() for k, v in (r.get("facings") or {}).items()
              if v and str(v).lower() in FACE}
    statements = []
    for a, words, b in r.get("statements") or []:
        statements.append((str(a).lower(), str(b).lower(), *_words(words)))
    a, b = (str(v).lower() for v in r["question"])
    return facing, statements, a, b


def decide(text: str) -> Optional[str]:
    """Parse -> entailed relation or UNKNOWN; None if the parse cannot be read or is inconsistent."""
    try:
        facing, statements, a, b = read_parse(text)
        names = {n for s, o, _, _ in statements for n in (s, o)} | {a, b}
        return entailed_relation({n: facing[n] for n in names if n in facing}, statements, a, b) or UNKNOWN
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------- template stories
FACING_RE = r"^(\w) (?:is facing|faces|is turned towards the) (\w+)\.$"


def parse_template(context: str):
    """facings and statements of a template story (used to compute gold labels from the generator's facts)."""
    facing, st = {}, []
    for s in re.split(r"(?<=\.)\s+", context):
        m = re.match(FACING_RE, s)
        if m:
            facing[m.group(1).lower()] = m.group(2)
            continue
        m2 = re.match(r"^(\w) is (?:on|to) (\w)'s", s)
        m = m2 or re.match(r"^(\w) is (.+?) (\w)(?:'s| and| ,|,|\.)", s)
        if not m:
            continue
        a = m.group(1).lower()
        b = (m2.group(2) if m2 else m.group(3)).lower()
        ahead = 1 if "in front of" in s else -1 if "behind" in s else 0
        right = 1 if re.search(r"\bright\b|right-hand", s) else -1 if re.search(r"\bleft\b|left-hand", s) else 0
        st.append((a, b, ahead, right))
    return facing, st


def gold_label(context: str, a: str, b: str) -> str:
    facing, st = parse_template(context)
    names = {n for s, o, _, _ in st for n in (s, o)} | {a, b}
    return entailed_relation({n: facing.get(n) for n in names}, st, a, b) or UNKNOWN


def path_references(statements: List[Statement], a: str, b: str):
    """Reference agents of the statements on the (BFS) path from b to a; None if they are not connected."""
    adj = defaultdict(list)
    for s, o, _, _ in statements:
        adj[s].append((o, o))
        adj[o].append((s, o))
    prev = {b: None}
    q = [b]
    while q:
        u = q.pop(0)
        for w, ref in adj[u]:
            if w not in prev:
                prev[w] = (u, ref)
                q.append(w)
    if a not in prev:
        return None
    refs, u = set(), a
    while prev[u]:
        u, ref = prev[u]
        refs.add(ref)
    return refs
