"""RoomSpace (Li, Hogg & Cohn, IJCAI 2024): exact grid CSP over all room layouts consistent with the text.

Semantics follow the benchmark's own reasoner (``solver.py`` in github.com/Fangjun-Li/RoomSpace), which
produced the gold answers. Every object is one cell of a D x D grid (D = 9 or 12; objects may share a cell):

    room regions  the nine thirds-by-thirds blocks, e.g. north-east: x >= 2D/3 and y >= 2D/3
    against wall  x or y on the border (x in {0, D-1} or y in {0, D-1})
    away walls    0 < x < D and 0 < y < D (as implemented by the reference reasoner)
    north of      same x, larger y; north-east of: larger x and larger y; same position: same cell
    distances     d2: short iff dist <= (D-1)/2; d3: short / moderate / far at 1/3 and 2/3 of the diagonal

"Could X be placed to the R of Y?" is Yes iff some consistent layout satisfies R. The grid size and the
distance granularity are not stated in the text, so a verdict is returned only when it holds under every
reading (D in {9, 12}; d2 and d3 unless "moderate" fixes d3; a painting hanging on the wall may carry either
wall fact). Under d3 the released answers admit more layouts than the words do, so Yes is proved with the
exact d3 thresholds and No with each word widened by one step. Each reading is solved exactly by search
with arc consistency; solutions and verdicts are cached inside the Scene.

Viewer-relative parses are converted to compass relations first (``ego_to_compass``): a viewer at the X wall's door
looking inwards faces away from X, and front is the viewer's forward direction (the benchmark's convention).
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ENTAILED, CONTRADICTED, UNKNOWN = "entailed", "contradicted", "unknown"
STATUSES = (ENTAILED, CONTRADICTED, UNKNOWN)


@dataclass
class Scene:
    """Formalized context. ``coverage`` is the fraction of context sentences turned into constraints."""

    coverage: float
    data: Any = None
    notes: Dict[str, Any] = field(default_factory=dict)


class Formalizer:
    """Dataset-family formalizer; subclasses implement the two methods below."""

    def scene(self, context: str, question: str) -> Scene:
        raise NotImplementedError

    def choice_status(self, scene: Scene, question: str, choices: Sequence[str], multi: bool) -> Dict[str, str]:
        """Status of every candidate answer (an atomic relation when ``multi``)."""
        raise NotImplementedError


# ----------------------------------------------------------------------------- viewer-relative words
ROT = {"north": 0, "east": 90, "south": 180, "west": 270}


COMP = {0: "north", 45: "north-east", 90: "east", 135: "south-east", 180: "south", 225: "south-west", 270: "west",
        315: "north-west"}


EGO_ANG = {"front": 0, "behind": 180, "left": 270, "right": 90}


def ego_to_compass(words, facing, front_is_forward=True):
    """words: set of egocentric words (front / behind / left / right); facing: viewer's compass facing."""
    vx = vy = 0.0
    for w in words:
        a = EGO_ANG[w] if (front_is_forward or w in ("left", "right")) else (EGO_ANG[w] + 180) % 360
        vx += math.sin(math.radians(a)); vy += math.cos(math.radians(a))
    if abs(vx) + abs(vy) < 1e-9:
        return None
    rel = round(math.degrees(math.atan2(vx, vy)) / 45) * 45 % 360
    comp = COMP[(rel + ROT[facing]) % 360]
    if HALF and "-" not in comp:
        comp += "-side"
    return comp


HALF = False


SIZES = (9, 12)
DIRECTIONS = {
    "north-east": "NE", "north-west": "NW", "south-east": "SE", "south-west": "SW",
    "north": "N", "south": "S", "east": "E", "west": "W",
}
REGIONS = {**DIRECTIONS, "center": "C"}
_DIR_RE = r"north-east|north-west|south-east|south-west|north|south|east|west"
_REG_RE = _DIR_RE + r"|center"

# A statement is a conjunction of atoms (obj1, obj2, direction keys); keys are ANDed.
Atom = Tuple[str, str, Tuple[str, ...]]


# ----------------------------------------------------------------------------- grid geometry


class _Grid:
    """Unary masks and binary relation matrices of one D x D grid (cell index = x * D + y)."""

    def __init__(self, d: int):
        self.d = d
        x, y = np.divmod(np.arange(d * d), d)
        lo, hi = d / 3, 2 * d / 3
        xs = {"W": x < lo, "M": (x >= lo) & (x < hi), "E": x >= hi}
        ys = {"S": y < lo, "M": (y >= lo) & (y < hi), "N": y >= hi}
        self.unary: Dict[str, np.ndarray] = {
            "N": xs["M"] & ys["N"], "S": xs["M"] & ys["S"], "E": xs["E"] & ys["M"], "W": xs["W"] & ys["M"],
            "C": xs["M"] & ys["M"], "NE": xs["E"] & ys["N"], "NW": xs["W"] & ys["N"],
            "SE": xs["E"] & ys["S"], "SW": xs["W"] & ys["S"],
            "TPP": (x == 0) | (x == d - 1) | (y == 0) | (y == d - 1),
            "NTPP": (x > 0) & (x < d) & (y > 0) & (y < d),
        }
        dx, dy = x[:, None] - x[None, :], y[:, None] - y[None, :]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        diag = np.sqrt(2 * (d - 1) ** 2)
        self.binary: Dict[str, np.ndarray] = {
            "N": (dx == 0) & (dy > 0), "S": (dx == 0) & (dy < 0), "E": (dx > 0) & (dy == 0),
            "W": (dx < 0) & (dy == 0), "NE": (dx > 0) & (dy > 0), "NW": (dx < 0) & (dy > 0),
            "SE": (dx > 0) & (dy < 0), "SW": (dx < 0) & (dy < 0), "O": (dx == 0) & (dy == 0),
            "CL3": dist <= diag / 3, "MD3": (dist > diag / 3) & (dist <= 2 * diag / 3), "FR3": dist > 2 * diag / 3,
            "CL2": dist <= (d - 1) / 2, "FR2": dist > (d - 1) / 2,
        }
        self.binary["CL3|MD3"] = self.binary["CL3"] | self.binary["MD3"]
        self.binary["MD3|FR3"] = self.binary["MD3"] | self.binary["FR3"]

    def mask(self, keys: Sequence[str]) -> np.ndarray:
        out = np.ones(self.d * self.d, dtype=bool)
        for k in keys:
            out = out & self.unary[k]
        return out

    def matrix(self, keys: Sequence[str]) -> np.ndarray:
        out = np.ones((self.d * self.d,) * 2, dtype=bool)
        for k in keys:
            out = out & self.binary[k]
        return out


@lru_cache(maxsize=None)
def _grid(d: int) -> _Grid:
    return _Grid(d)


_DIST = {"d2": {"short": "CL2", "far": "FR2"}, "d3": {"short": "CL3", "moderate": "MD3", "far": "FR3"}}
# The released d3 answers admit more layouts than the d3 words do (a word may stand for the next coarser
# fact); this one-step widening covers them. Possibility (Yes) uses _DIST, impossibility (No) this table.
_DIST_LOOSE = {"d3": {"short": "CL3|MD3", "moderate": "MD3|FR3", "far": "FR3"}}


# ----------------------------------------------------------------------------- exact CSP


class _CSP:
    """One reading of the scene: exact satisfiability by domain splitting with arc consistency."""

    def __init__(self, grid: _Grid, n: int, unary: Dict[int, Tuple[str, ...]], binary: List[Tuple[int, int, tuple]]):
        self.grid, self.n = grid, n
        self.base = [grid.mask(unary.get(i, ())) for i in range(n)]
        self.cons = [(i, j, grid.matrix(k)) for i, j, k in binary]
        self.pool: List[np.ndarray] = []
        self.cache: Dict[tuple, bool] = {}

    def sat(self, key: tuple, extra: Tuple[Tuple[int, int, np.ndarray], ...]) -> bool:
        """Is the scene plus the resolved constraints ``extra`` (named by ``key``) satisfiable?"""
        if key in self.cache:
            return self.cache[key]
        doms = list(self.base)
        cons = self.cons + list(extra)
        ok = any(self._holds(s, extra) for s in self.pool)
        if not ok:
            merged: Dict[Tuple[int, int], np.ndarray] = {}
            for i, j, m in cons:  # one matrix per pair, so arc consistency sees their conjunction
                if i == j:
                    doms[i] = doms[i] & np.diagonal(m)
                    continue
                i, j, m = (i, j, m) if i < j else (j, i, m.T)
                merged[i, j] = merged[i, j] & m if (i, j) in merged else m
            mats = [(i, j, m.astype(np.float32), m.T.astype(np.float32)) for (i, j), m in merged.items()]
            sol = self._search(doms, mats)
            if sol is not None:
                self.pool.append(sol)
                ok = True
        self.cache[key] = ok
        return ok

    @staticmethod
    def _holds(sol: np.ndarray, extra: Tuple[Tuple[int, int, np.ndarray], ...]) -> bool:
        return all(m[sol[i], sol[j]] for i, j, m in extra)

    @staticmethod
    def _propagate(doms: List[np.ndarray], mats) -> bool:
        changed = True
        while changed:
            changed = False
            for i, j, m, mt in mats:
                di = doms[i] & (m @ doms[j].astype(np.float32) > 0)
                if di.sum() != doms[i].sum():
                    if not di.any():
                        return False
                    doms[i], changed = di, True
                dj = doms[j] & (mt @ doms[i].astype(np.float32) > 0)
                if dj.sum() != doms[j].sum():
                    if not dj.any():
                        return False
                    doms[j], changed = dj, True
        return all(d.any() for d in doms)

    def _search(self, doms: List[np.ndarray], mats) -> Optional[np.ndarray]:
        doms = [d.copy() for d in doms]
        if not self._propagate(doms, mats):
            return None
        sizes = [int(d.sum()) for d in doms]
        if max(sizes, default=1) == 1:
            return np.array([int(np.flatnonzero(d)[0]) for d in doms])
        i = min((s, k) for k, s in enumerate(sizes) if s > 1)[1]
        idx = np.flatnonzero(doms[i])
        for part in (idx[: len(idx) // 2], idx[len(idx) // 2:]):
            sub = list(doms)
            sub[i] = np.zeros_like(doms[i])
            sub[i][part] = True
            sol = self._search(sub, mats)
            if sol is not None:
                return sol
        return None


# ----------------------------------------------------------------------------- scene text


def _norm(name: str) -> str:
    name = " ".join(name.lower().strip(" .,;").split())
    return re.sub(r"^(?:the|a|an) ", "", name)


@dataclass
class _Facts:
    objects: List[str] = field(default_factory=list)
    unary: List[Tuple[str, str]] = field(default_factory=list)  # (obj, region key | TPP | NTPP | WALL)
    binary: List[Tuple[str, str, str]] = field(default_factory=list)  # (obj1, direction key, obj2)
    dist: List[Tuple[str, str, str]] = field(default_factory=list)  # (obj1, short|moderate|far, obj2)

    def obj(self, name: str) -> str:
        name = _norm(name)
        if name not in self.objects:
            self.objects.append(name)
        return name


_ROOM = re.compile(r"^imagine a square-shaped (?P<room>[\w ]+), bordered by four walls$")
_LIST = re.compile(r"^this room contains a collection of furniture, including (.+)$")
_ITEM = re.compile(
    rf"^(?P<name>[\w ]+?)(?: placed in the (?P<reg>{_REG_RE})| hanging on the wall,? in the (?P<wreg>{_REG_RE})"
    rf" of the (?:the )?room| in the (?P<preg>{_REG_RE}) of the (?:the )?room)?"
    r"(?:, (?P<tpp>against the wall|away from walls))?$"
)
_PAIR = re.compile(
    rf"^(?P<a>[\w ]+?) is placed (?:to the (?P<rel>{_DIR_RE}) of|to the at the same position of|at the same position"
    rf" as) (?P<bs>[\w ]+?)(?:, at a (?P<dist>short|moderate|far) distance)?$"
)


def _list_items(text: str) -> List[str]:
    out: List[str] = []
    for frag in re.split(r"[;,] ", text):
        frag = frag.strip(" .;,")
        if out and re.match(r"^(?:against the wall|away from walls|in the [\w-]+ of the (?:the )?room)$", frag):
            out[-1] += ", " + frag
        elif frag:
            out.append(frag)
    return out


def _parse_list(text: str, facts: _Facts) -> bool:
    for item in _list_items(text):
        item = re.sub(r"^(?:and )?some [\w ]+?(?: hanging on the wall)?: ", "", item)
        item = re.sub(r"^and ", "", item)
        m = _ITEM.match(item)
        if not m or re.search(r"\b(?:placed|in the|hanging)\b", m["name"]):
            return False
        name = facts.obj(m["name"])
        reg = m["reg"] or m["wreg"] or m["preg"]
        if reg:
            facts.unary.append((name, REGIONS[reg]))
        if m["tpp"]:
            facts.unary.append((name, "TPP" if m["tpp"] == "against the wall" else "NTPP"))
        elif m["wreg"] or m["preg"]:
            facts.unary.append((name, "WALL"))
    return True


def _sentences(text: str) -> List[str]:
    return [s for s in (x.strip() for x in re.split(r"(?<=\.)\s+", text.replace("\n", " "))) if s]


def _parse_context(context: str) -> Tuple[_Facts, int, int]:
    facts, parsed, sentences = _Facts(), 0, _sentences(context)
    for s in sentences:
        s = s.lower().rstrip(".").strip()
        m = _ROOM.match(s)
        if m:
            parsed += 1
            continue
        m = _LIST.match(s)
        if m:
            parsed += _parse_list(m[1], facts)
            continue
        m = _PAIR.match(s)
        if m:
            a = facts.obj(m["a"])
            for b in re.split(r" and ", m["bs"]):
                b = facts.obj(b)
                facts.binary.append((a, DIRECTIONS[m["rel"]] if m["rel"] else "O", b))
                if m["dist"]:
                    facts.dist.append((a, m["dist"], b))
            parsed += 1
    return facts, parsed, len(sentences)


@dataclass
class _Reading:
    tight: _CSP  # layouts certainly admitted by the reference facts
    loose: _CSP  # layouts that cover every layout admitted by the reference facts


class _World:
    """All readings of one scene; statements are decided only when every reading agrees."""

    def __init__(self, facts: _Facts):
        self.facts = facts
        self.index = {o: k for k, o in enumerate(facts.objects)}
        dist_modes: List[Optional[str]] = [None]
        if facts.dist:
            dist_modes = ["d3"] if any(w == "moderate" for _, w, _ in facts.dist) else ["d2", "d3"]
        walls = [o for o, k in facts.unary if k == "WALL"]
        self.specs = [(d, mode, dict(zip(walls, w))) for d in SIZES for mode in dist_modes
                      for w in itertools.product(("TPP", "NTPP"), repeat=len(walls))]
        self._readings: Dict[int, _Reading] = {}

    def reading(self, k: int) -> _Reading:
        if k not in self._readings:
            d, mode, walls = self.specs[k]
            unary: Dict[int, Tuple[str, ...]] = {}
            for o, key in self.facts.unary:
                key = walls.get(o, key) if key == "WALL" else key
                unary[self.index[o]] = unary.get(self.index[o], ()) + (key,)
            binary = [(self.index[a], self.index[b], (r,)) for a, r, b in self.facts.binary]

            def csp(table: Dict[str, Dict[str, str]]) -> _CSP:
                dist = [(self.index[a], self.index[b], (table[mode][w],)) for a, w, b in self.facts.dist]
                return _CSP(_grid(d), len(self.index), unary, binary + dist)

            tight = csp(_DIST)
            loose = csp(_DIST_LOOSE) if mode == "d3" else tight
            self._readings[k] = _Reading(tight, loose)
        return self._readings[k]

    def status(self, atoms: Sequence[Atom]) -> str:
        """Possibility: entailed if some layout satisfies the atoms, contradicted if none does, under every reading."""
        if any(a not in self.index or b not in self.index for a, b, _ in atoms):
            return UNKNOWN
        verdict = None
        for k in range(len(self.specs)):
            v = self._decide(self.reading(k), tuple(sorted(atoms)))
            if v == UNKNOWN or (verdict is not None and v != verdict):
                return UNKNOWN
            verdict = v
        return verdict or UNKNOWN

    def _sat(self, rd: _Reading, atoms: Tuple[Atom, ...], loose: bool) -> bool:
        extra = tuple((self.index[a], self.index[b], rd.tight.grid.matrix(keys)) for a, b, keys in atoms)
        return (rd.loose if loose else rd.tight).sat((loose,) + atoms, extra)

    def _decide(self, rd: _Reading, atoms: Tuple[Atom, ...]) -> str:
        if not self._sat(rd, atoms, loose=True):
            return CONTRADICTED
        return ENTAILED if self._sat(rd, atoms, loose=False) else UNKNOWN


# ----------------------------------------------------------------------------- questions


_QUESTION = re.compile(
    rf"^could (?P<a>[\w ]+?) be placed (?:to the (?P<rel>{_DIR_RE}) of|in the same location as) (?P<b>[\w ]+?)\?$"
)


def _flip(status: str) -> str:
    return {ENTAILED: CONTRADICTED, CONTRADICTED: ENTAILED}.get(status, status)


class RoomSpaceFormalizer(Formalizer):
    def scene(self, context: str, question: str) -> Scene:
        facts, parsed, total = _parse_context(context)
        world = _World(facts) if parsed == total and facts.objects else None
        return Scene(coverage=parsed / max(total, 1), data=world)

    def choice_status(self, scene: Scene, question: str, choices: Sequence[str], multi: bool) -> Dict[str, str]:
        out = {c: UNKNOWN for c in choices}
        m = _QUESTION.match(" ".join(question.lower().split()))
        world: Optional[_World] = scene.data
        if not m or world is None:
            return out
        rel = DIRECTIONS[m["rel"]] if m["rel"] else "O"
        yes = world.status([(_norm(m["a"]), _norm(m["b"]), (rel,))])
        for c in choices:
            if c.lower() == "yes":
                out[c] = yes
            elif c.lower() == "no":
                out[c] = _flip(yes)
        return out
