"""SpatialEval Maze-Nav (Wang et al., 2024), text-only: 1,000 of the 1,500 questions (seed 1). Questions ask for the
number of right / left / total turns along the marked path, or whether the exit lies in a direction from the start."""

import json
import random
import re

from enfor.env import DATASETS_DIR
from enfor.solvers import Undecided, maze
from enfor.tasks.common import COT_SYSTEM, SPEC, parse_record, read_cot as _read_cot, user_text

COLUMNS = {"spatialeval_maze": "maze"}
COL = "spatialeval_maze"


def items(column):
    import pandas as pd

    d = pd.read_parquet(DATASETS_DIR / "SpatialEval" / "tqa" / "test-00000-of-00001.parquet")
    rows = [r for r in d.itertuples() if r.id.startswith("mazenav.")]
    rows = sorted(random.Random(1).sample(rows, 1000), key=lambda r: r.id)
    out = []
    for r in rows:
        body, opts = r.text.split("Available options:")
        options = dict(re.findall(r"^\s*([A-D])\. (.*?)\.?$", opts.strip(), re.M))
        out.append(dict(key=f"semaze{r.id}", context=body.strip(), question=None, options=options,
                        gold=r.oracle_option, meta={"id": r.id, "qtype": r.id.split(".")[-1]}))
    return out


def gold(column, x):
    return x["gold"]


# ---------------------------------------------------------------------------------------------- parser
PARSER_SYSTEM = 'Transcribe the maze and the question for a solver; do not answer and do not trace the path. Reply with only a JSON object {"grid": [rows], "question": Q}: grid copies every row of the ASCII maze exactly, character by character (keep spaces). Q is {"kind": "right_turns"}, {"kind": "left_turns"} or {"kind": "total_turns"} for questions about the number of turns, or {"kind": "position", "relation": R} for \'Is the exit (E) <R> the starting point (S)?\', with R copied as written (for example "directly to the left of", "to the bottom right of").'
PARSER_SHOT_USER = 'Here is a Maze represented in ASCII code:\n######\n#SXX #\n## X #\n#  XE#\n######, where the symbols have the following meanings: (as above: # wall, space open, S start, E exit, X the route).\nHow many right turns are there in the provided path (marked by X) from S to E?\nAvailable options:\nA. 3\nB. 1\nC. 0\nD. 2'
PARSER_SHOT = {'grid': ['######', '#SXX #', '## X #', '#  XE#', '######'], 'question': {'kind': 'right_turns'}}
TAIL = "Reply with only the JSON record."


def parser_prompt(column, X):
    prefix = [{"role": "user", "content": f"{PARSER_SHOT_USER}\n{TAIL}"},
              {"role": "assistant", "content": json.dumps(PARSER_SHOT, ensure_ascii=False)}]
    return dict(system=PARSER_SYSTEM, prefix=prefix, effort="low", max_tokens=8192,
                requests=[(x["key"], user_text(COL, x).replace(SPEC[COL], TAIL)) for x in X])


def cot_prompt(column, X):
    return dict(system=COT_SYSTEM, prefix=[], requests=[(x["key"], user_text(COL, x)) for x in X], effort="low",
                max_tokens=8192)


def decide(column, text, x):
    try:
        rec = parse_record(text)
        if rec is None:
            return None
        q = rec.get("question") or {}
        if "grid" in rec:
            moves = maze.moves_of(maze.maze_path(rec["grid"]))
        else:
            moves = [maze.MOVE[str(m).strip().lower()] for m in rec["moves"]]
            if not moves:
                raise Undecided("no moves")
        if q.get("kind") == "position" and not q.get("relation"):
            raise Undecided("relation")
        return maze.maze_answer(moves, q, x["options"])
    except Exception:
        return None


def read_cot(column, text, x):
    return _read_cot(COL, text, x)
