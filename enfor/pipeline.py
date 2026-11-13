"""EnFoR end to end: parse every item (k samples), decide each parse with the exact solver, take the majority over the
decided parses, and fall back to the same model's CoT answer when no parse is decided.

Logs: <ENFOR_OUTPUT>/enfor/<model>/<source>.parse.jsonl and <column>.cot.jsonl (resumable).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from enfor import llm
from enfor.env import OUTPUT_DIR
from enfor.tasks import SOURCES, metric, task

RUNS = OUTPUT_DIR / "enfor"


def source_of(column: str) -> str:
    return next((s for s, cols in SOURCES.items() if column in cols), column)


def source_items(source: str) -> List[dict]:
    cols = SOURCES.get(source, [source])
    seen, out = set(), []
    for c in cols:
        for x in task(c).items(c):
            if x["key"] not in seen:
                seen.add(x["key"])
                out.append(x)
    return out


def parse_key(key: str, j: int, k: int) -> str:
    """Log key of parse j of an item: the item key for a single parse, <key>#p<j> for k parses."""
    return key if k == 1 else f"{key}#p{j}"


def run_parser(column: str, model: str, k: int = 1, runs: Path = RUNS) -> Path:
    """k parses per item in one sample-major run: parse j of item i has seed j * n + i, so parse 0 is the single parse."""
    source = source_of(column)
    X = source_items(source)
    p = task(column).parser_prompt(column, X)
    requests = [(parse_key(key, j, k), user) for j in range(k) for key, user in p["requests"]]
    log = runs / model / f"{source}.parse.jsonl"
    llm.generate(model, p["system"], p["prefix"], requests, log, effort=p["effort"], max_tokens=p["max_tokens"])
    return log


def run_cot(column: str, model: str, runs: Path = RUNS) -> Path:
    X = task(column).items(column)
    p = task(column).cot_prompt(column, X)
    log = runs / model / f"{column}.cot.jsonl"
    llm.generate(model, p["system"], p["prefix"], p["requests"], log, effort=p["effort"], max_tokens=p["max_tokens"])
    return log


def vote(outs: List[Optional[object]]):
    """Majority over the decided outputs (None = undecided); ties go to the earliest parse among the tied."""
    c = Counter(o for o in outs if o is not None)
    if not c:
        return None
    top = max(c.values())
    return next(o for o in outs if o is not None and c[o] == top)


def evaluate(column: str, parse_log: Path, cot_log: Optional[Path], k: int = 1) -> Dict[str, float]:
    """value (with CoT fallback), value without fallback, and the share of items the solver decides."""
    t = task(column)
    X = t.items(column)
    parses = {key: r["text"] for key, r in llm.read_log(parse_log).items()}
    cots = {key: r["text"] for key, r in llm.read_log(cot_log).items()} if cot_log else {}
    decided, final = {}, {}
    for x in X:
        outs = [t.decide(column, parses.get(parse_key(x["key"], j, k), parses.get(x["key"]) if j == 0 else None), x)
                for j in range(k)]
        d = vote(outs)
        decided[x["key"]] = d
        final[x["key"]] = d if d is not None else t.read_cot(column, cots.get(x["key"]), x)
    return {"value": metric(column, X, final), "value_nofallback": metric(column, X, decided),
            "coverage": sum(v is not None for v in decided.values()) / len(X), "n": len(X)}
