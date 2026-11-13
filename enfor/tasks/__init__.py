"""Task registry: one module per benchmark family, one column per test set of the paper's main table.

Every task module provides ``items(column)``, ``gold(column, x)``, ``parser_prompt(column, X)``,
``decide(column, text, x)`` (parse -> the solver's answer, or None when the parse cannot be read or the solver leaves
the question undecided), ``cot_prompt(column, X)`` and ``read_cot(column, text, x)`` for the CoT fallback, and
optionally ``metric(column, X, pred)`` (default: accuracy).
"""

from importlib import import_module

FAMILIES = ["stepgame", "forest", "route_survey", "roomspace", "maze", "spacenli"]
COLUMNS = {}
for _family in FAMILIES:
    _module = import_module(f"enfor.tasks.{_family}")
    for _column in _module.COLUMNS:
        COLUMNS[_column] = _module

# Columns parsed together in one run (one parse per item serves every column of its source).
SOURCES = {"forest": ["forest_amb_f1", "forest_stated"], "route": ["route_rm", "route_mr"]}


def task(column):
    return COLUMNS[column]


def metric(column, X, pred):
    module = COLUMNS[column]
    if hasattr(module, "metric"):
        return module.metric(column, X, pred)
    return sum(pred.get(x["key"]) == module.gold(column, x) for x in X) / len(X)
