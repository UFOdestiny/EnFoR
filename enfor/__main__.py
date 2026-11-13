"""Command line: python -m enfor <command> ...

    python -m enfor list
    python -m enfor parse    --column sg_heldout --model gpt-oss-120b [--k 5]
    python -m enfor cot      --column sg_heldout --model gpt-oss-120b
    python -m enfor evaluate --column sg_heldout --model gpt-oss-120b [--k 5]
    python -m enfor run      --column sg_heldout --model gpt-oss-120b [--k 5]     # parse + cot + evaluate
"""

import argparse
import json

from enfor import pipeline
from enfor.tasks import COLUMNS


def main():
    ap = argparse.ArgumentParser(prog="python -m enfor")
    ap.add_argument("command", choices=["list", "parse", "cot", "evaluate", "run"])
    ap.add_argument("--column", choices=list(COLUMNS))
    ap.add_argument("--model", help="a folder under ENFOR_MODELS (vLLM) or a Claude model name")
    ap.add_argument("--k", type=int, default=1, help="parses per item; the answer is the majority of decided parses")
    a = ap.parse_args()
    if a.command == "list":
        print("\n".join(COLUMNS))
        return
    if not (a.column and a.model):
        ap.error("--column and --model are required")
    parse_log = pipeline.RUNS / a.model / f"{pipeline.source_of(a.column)}.parse.jsonl"
    cot_log = pipeline.RUNS / a.model / f"{a.column}.cot.jsonl"
    if a.command in ("parse", "run"):
        parse_log = pipeline.run_parser(a.column, a.model, k=a.k)
    if a.command in ("cot", "run"):
        cot_log = pipeline.run_cot(a.column, a.model)
    if a.command in ("evaluate", "run"):
        r = pipeline.evaluate(a.column, parse_log, cot_log if cot_log.exists() else None, k=a.k)
        print(json.dumps({"column": a.column, "model": a.model, "k": a.k, **r}, indent=1))


if __name__ == "__main__":
    main()
