"""Maze navigation (SpatialEval Maze-Nav): the parse copies the ASCII grid; the solver traces the unique S -> E path over
the X cells and counts turns relative to the traveler's heading, or reads the exit's position relative to the start."""

import re

from enfor.solvers import Undecided

MOVE = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def maze_path(grid):
    """Unique cell path S -> X... -> E over 4-neighbours (Undecided if absent or branching)."""
    rows = [str(r) for r in grid]
    cells = {(i, j): ch for i, r in enumerate(rows) for j, ch in enumerate(r)}
    S = [p for p, c in cells.items() if c == "S"]; E = [p for p, c in cells.items() if c == "E"]
    if len(S) != 1 or len(E) != 1:
        raise Undecided("S/E")
    S, E = S[0], E[0]
    route = {p for p, c in cells.items() if c in "XSE"}
    path = [S]; prev = None
    while path[-1] != E:
        u = path[-1]
        nb = [(u[0] + d[0], u[1] + d[1]) for d in MOVE.values() if (u[0] + d[0], u[1] + d[1]) in route
              and (u[0] + d[0], u[1] + d[1]) != prev and (u[0] + d[0], u[1] + d[1]) not in path]
        if E in nb:
            nb = [E]
        if len(nb) != 1:
            raise Undecided("path not unique")
        prev = u; path.append(nb[0])
        if len(path) > 400:
            raise Undecided("loop")
    return path


def moves_of(path):
    return [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]


def maze_answer(moves, question, options):
    """moves: list of (drow, dcol) unit steps; question: {"kind": "right_turns" | "left_turns" | "total_turns" |
    "position", "relation": ...}."""
    kind = question.get("kind")
    turns = list(zip(moves, moves[1:]))
    if kind in ("right_turns", "left_turns", "total_turns"):
        r = sum(1 for u, v in turns if v == (u[1], -u[0]))       # clockwise on screen (row down, col right)
        l = sum(1 for u, v in turns if v == (-u[1], u[0]))
        val = {"right_turns": r, "left_turns": l, "total_turns": r + l}[kind]
        for k, v in options.items():
            if str(v).strip() == str(val):
                return k
        raise Undecided("value not among options")
    if kind == "position":
        dr = sum(m[0] for m in moves); dc = sum(m[1] for m in moves)
        rel = str(question.get("relation", "")).lower()
        tests = {"directly left": dr == 0 and dc < 0, "directly right": dr == 0 and dc > 0,
                 "directly above": dc == 0 and dr < 0, "directly below": dc == 0 and dr > 0,
                 "top left": dr < 0 and dc < 0, "top right": dr < 0 and dc > 0,
                 "bottom left": dr > 0 and dc < 0, "bottom right": dr > 0 and dc > 0}
        rel = re.sub(r"\b(to|the|of|side)\b", " ", rel.replace("-", " ").replace("_", " "))
        rel = re.sub(r"\s+", " ", rel).strip()
        key = next((k for k in tests if k in rel), None)
        if key is None:
            raise Undecided("bad relation")
        want = "yes" if tests[key] else "no"
        for k, v in options.items():
            if str(v).strip().lower() == want:
                return k
        raise Undecided("yes/no not among options")
    raise Undecided("bad kind")

