"""Spatial natural language inference (SpaceNLI): the parse writes premise and hypothesis as atoms in the frames they
are stated in; the solver encodes them over before / during / after in Z3 (containment, motion, viewer-relative and
intrinsic directions) and returns entailment if the premises entail the hypothesis, contradiction if they entail its
negation, and neutral if both are satisfiable."""

import itertools
import re

from enfor.solvers import Undecided as _Undecided

STATIC = {"in", "outside", "at", "on", "on_top_of", "under", "above", "below", "in_front_of", "behind", "left_of",
          "right_of", "between", "near", "far_from", "with", "different", "north_of", "south_of", "east_of", "west_of"}


MOTION = {"into", "out_of", "from", "to", "through", "across", "towards", "away_from", "by"}


NLSAT = True


class Undecided(_Undecided):
    pass


def nm(s):
    s = re.sub(r"^(the|a|an)\s+", "", str(s).strip().lower())
    return re.sub(r"\s+", " ", s.replace("'s", "")).strip()


def clean_atom(a):
    if not isinstance(a, dict):
        raise Undecided("bad atom")
    rel = str(a.get("rel", "")).strip().lower().replace(" ", "_").replace("-", "_")
    rel = {"inside": "in", "within": "in", "on_top": "on_top_of", "in_front": "in_front_of", "front": "in_front_of",
           "close_to": "near", "far": "far_from", "toward": "towards", "away": "away_from", "enter": "into",
           "leave": "out_of", "past": "by", "left": "left_of", "right": "right_of", "north": "north_of",
           "south": "south_of", "east": "east_of", "west": "west_of", "beneath": "under", "underneath": "under",
           "over": "above"}.get(rel, rel)
    if rel not in STATIC | MOTION:
        raise Undecided(f"unsupported {rel}")
    args = a.get("args") or []
    need = 3 if rel == "between" else 2
    core = [nm(x) for x in args[:need] if x is not None]
    if len(core) < need:
        raise Undecided("args")
    viewer = nm(args[2]) if rel in ("in_front_of", "behind", "left_of", "right_of") and len(args) > 2 and args[2] else None
    t = str(a.get("time") or "now").lower()
    t = t if t in ("now", "past", "before") else "now"
    asp = str(a.get("aspect") or "completed").lower()
    asp = "progressive" if asp.startswith("prog") else "completed"
    return dict(rel=rel, args=core, viewer=viewer, neg=bool(a.get("neg")), time=t, aspect=asp)


def decide(rec, timeout_ms=8000):
    """'entailment' | 'contradiction' | 'neutral'; Undecided otherwise."""
    import z3
    if not isinstance(rec, dict):
        raise Undecided("no record")
    P = [clean_atom(a) for a in rec.get("premises") or []]
    H = [clean_atom(a) for a in rec.get("hypothesis") or []]
    if not P or not H:
        raise Undecided("empty")
    # proper-noun places only (capitalised as written): the closed world applies to cities / countries
    named = {nm(n) for n in rec.get("named_places") or [] if str(n).strip()[:1].isupper()}
    ents = sorted({e for a in P + H for e in a["args"] + ([a["viewer"]] if a["viewer"] else [])} | {"_viewer"})
    T = ["before", "during", "after"]
    prog = any(a["rel"] in MOTION and a["aspect"] == "progressive" for a in P)
    comp = any(a["rel"] in MOTION and a["aspect"] == "completed" for a in P)
    now = "during" if prog or not comp else "after"
    # without a motion event, past-tense statements describe one situation (the 'now' of the description)
    upto = T[:T.index(now) + 1] if (prog or comp) else [now]
    s = z3.Tactic("qfnra-nlsat").solver() if NLSAT else z3.Solver(); s.set("timeout", timeout_ms)
    B = {}

    def top(p, a, b, t):
        k = (p, a, b, t)
        if k not in B:
            B[k] = z3.Bool("%s|%s|%s|%s" % k)
        return B[k]
    DJ = {}

    def disj(a, b):
        a, b = sorted((a, b))
        if (a, b) not in DJ:
            DJ[(a, b)] = z3.Bool(f"dj|{a}|{b}")
        return DJ[(a, b)]
    EV = {}

    def ev(k, a, b):
        if (k, a, b) not in EV:
            EV[(k, a, b)] = z3.Bool(f"ev|{k}|{a}|{b}")
        return EV[(k, a, b)]
    X = {e: z3.Real("x|" + e) for e in ents}; Y = {e: z3.Real("y|" + e) for e in ents}
    Z = {e: z3.Real("z|" + e) for e in ents}
    FX = {e: z3.Real("fx|" + e) for e in ents}; FY = {e: z3.Real("fy|" + e) for e in ents}
    for e in ents:
        s.add(z3.Or(z3.And(FX[e] == 0, FY[e] == 1), z3.And(FX[e] == 0, FY[e] == -1),
                    z3.And(FX[e] == 1, FY[e] == 0), z3.And(FX[e] == -1, FY[e] == 0)))
    # axioms
    for t in T:
        for a in ents:
            s.add(z3.Not(top("in", a, a, t)), z3.Not(top("at", a, a, t)), z3.Not(top("on", a, a, t)))
        for a, b in itertools.permutations(ents, 2):
            s.add(z3.Implies(top("in", a, b, t), top("at", a, b, t)), z3.Implies(top("on", a, b, t), top("at", a, b, t)))
            s.add(z3.Implies(top("in", a, b, t), z3.Not(top("in", b, a, t))))
            s.add(top("with", a, b, t) == top("with", b, a, t))
            s.add(z3.Implies(top("in", a, b, t), z3.Not(top("on", a, b, t))))
            s.add(z3.Implies(top("in", a, b, t), z3.Not(disj(a, b))))
        for a, b, c in itertools.permutations(ents, 3):
            s.add(z3.Implies(z3.And(top("in", a, b, t), top("in", b, c, t)), top("in", a, c, t)))
            s.add(z3.Implies(z3.And(top("at", a, b, t), top("in", b, c, t)), top("at", a, c, t)))
            s.add(z3.Implies(z3.And(top("at", a, b, t), disj(b, c)), z3.Not(top("at", a, c, t))))
    for a, b, c in itertools.permutations(ents, 3):
        for k in ("from", "to"):
            s.add(z3.Implies(z3.And(ev(k, a, b), top("in", b, c, "during")), ev(k, a, c)))
    for a, b in itertools.permutations(ents, 2):
        s.add(z3.Not(z3.And(ev("towards", a, b), ev("away_from", a, b))))
        s.add(z3.Not(z3.And(ev("into", a, b), ev("out_of", a, b))))
        s.add(z3.Implies(top("on", a, b, now), Z[a] > Z[b]))
        s.add(z3.Implies(top("in", a, b, now), Z[a] == Z[b]))
    # closed world for named places: no stated containment -> disjoint
    stated = {(a["args"][0], a["args"][1]) for a in P if a["rel"] == "in" and not a["neg"]}
    for a, b in itertools.combinations(sorted(named & set(ents)), 2):
        if not _linked(a, b, stated):
            s.add(disj(a, b))

    def dist2(a, b):
        return (X[a] - X[b]) ** 2 + (Y[a] - Y[b]) ** 2

    def static_formula(at, times):
        """Formula of a state atom; times: list of time points (conjunction for premises)."""
        r, g = at["rel"], at["args"]
        a, b = g[0], g[1]
        if r in ("in", "at", "on", "with", "outside"):
            f = top("in" if r == "outside" else r, a, b, times[0])
            return z3.Not(f) if r == "outside" else f
        if r == "different":
            return disj(a, b)
        if r in ("above", "on_top_of"):
            f = Z[a] > Z[b]
            return z3.And(f, top("on", a, b, now)) if r == "on_top_of" else f
        if r in ("below", "under"):
            return Z[a] < Z[b]
        if r in ("north_of", "south_of", "east_of", "west_of"):
            return {"north_of": Y[a] > Y[b], "south_of": Y[a] < Y[b], "east_of": X[a] > X[b], "west_of": X[a] < X[b]}[r]
        if r in ("in_front_of", "behind"):
            v = at["viewer"]
            if v:
                d = (X[a] - X[b]) * (X[v] - X[b]) + (Y[a] - Y[b]) * (Y[v] - Y[b])
            else:
                d = (X[a] - X[b]) * FX[b] + (Y[a] - Y[b]) * FY[b]
            return d > 0 if r == "in_front_of" else d < 0
        if r in ("left_of", "right_of"):
            v = at["viewer"] or "_viewer"
            cr = (X[b] - X[v]) * (Y[a] - Y[v]) - (Y[b] - Y[v]) * (X[a] - X[v])
            return z3.And(cr > 0 if r == "left_of" else cr < 0, (X[b] - X[v]) ** 2 + (Y[b] - Y[v]) ** 2 > 0)
        if r == "between":
            # A strictly inside segment BC (horizontal plane), without an existential parameter:
            # collinear and (B - A) . (C - A) < 0
            c = g[2]
            cross = (X[b] - X[a]) * (Y[c] - Y[a]) - (Y[b] - Y[a]) * (X[c] - X[a])
            dot = (X[b] - X[a]) * (X[c] - X[a]) + (Y[b] - Y[a]) * (Y[c] - Y[a])
            return z3.And(cross == 0, dot < 0)
        if r == "near":
            return dist2(a, b) <= 1
        if r == "far_from":
            return dist2(a, b) >= 100
        raise Undecided(r)

    def motion_consequences(at):
        r, (a, b), asp = at["rel"], at["args"], at["aspect"]
        done = asp == "completed"
        cs = []
        if r == "into":
            cs.append(z3.Not(top("in", a, b, "before")))
            if done:
                cs.append(top("in", a, b, "after"))
        elif r == "out_of":
            cs.append(top("in", a, b, "before"))
            if done:
                cs.append(z3.Not(top("in", a, b, "after")))
        elif r == "from":
            cs.append(top("at", a, b, "before"))
        elif r == "to":
            if done:
                cs.append(top("at", a, b, "after"))
        elif r == "through":
            cs.append(top("in", a, b, "during"))
        elif r == "across":
            cs.append(top("at", a, b, "during"))
        elif r == "towards":
            if not done:
                cs += [z3.Not(top(p, a, b, "during")) for p in ("at", "on", "in")]
        elif r == "away_from":
            if done:
                cs += [z3.Not(top(p, a, b, "after")) for p in ("at", "on", "in", "with")]
        return cs

    movers = {at["args"][0] for at in P if at["rel"] in MOTION}
    # premises
    for at in P:
        if at["rel"] in MOTION:
            e = ev(at["rel"], *at["args"])
            if at["neg"]:
                s.add(z3.Not(e)); continue
            s.add(e)
            for c in motion_consequences(at):
                s.add(c)
            continue
        if at["rel"] in ("in", "at", "on", "with", "outside"):
            if at["time"] == "past":
                f = z3.Or(*[static_formula(at, [t]) for t in upto])
            elif at["time"] == "before":
                f = static_formula(at, ["before"])
            elif at["args"][0] not in movers:          # a static fact about a non-moving entity holds throughout
                f = z3.And(*[static_formula(at, [t]) for t in T])
            else:
                f = static_formula(at, [now])
        else:
            f = static_formula(at, [now])
        if at["rel"] in ("in_front_of", "behind", "left_of", "right_of", "under", "below") and not at["neg"]:
            s.add(z3.Not(top("in", at["args"][0], at["args"][1], now)))   # placed beside / under B, so not in B
        if at["neg"]:
            s.add(z3.Not(f))
            if at["rel"] == "in" and at["time"] == "now" and at["args"][0] not in movers:
                s.add(disj(*at["args"]))
        else:
            s.add(f)
    # hypothesis
    hs = []
    for at in H:
        if at["rel"] in MOTION:
            f = ev(at["rel"], *at["args"])
        elif at["rel"] in ("in", "at", "on", "with", "outside"):
            if at["time"] == "past":
                f = z3.Or(*[static_formula(at, [t]) for t in upto]) if not at["neg"] else \
                    z3.Or(*[z3.Not(static_formula(at, [t])) for t in upto])
                hs.append(f); continue
            f = static_formula(at, ["before" if at["time"] == "before" else now])
        else:
            f = static_formula(at, [now])
        hs.append(z3.Not(f) if at["neg"] else f)
    h = z3.And(*hs)
    # consequences of every event atom, including event atoms derived by monotonicity or asked by the hypothesis
    asp = "completed" if comp and not prog else "progressive"
    for (k, a, b), e in list(EV.items()):
        for c in motion_consequences(dict(rel=k, args=[a, b], aspect=asp)):
            s.add(z3.Implies(e, c))
    r0 = s.check()
    if r0 != z3.sat:
        raise Undecided("premises inconsistent" if r0 == z3.unsat else "timeout")
    s.push(); s.add(z3.Not(h)); r1 = s.check(); s.pop()
    if r1 == z3.unsat:
        return "entailment"
    s.push(); s.add(h); r2 = s.check(); s.pop()
    if r2 == z3.unsat:
        return "contradiction"
    if r1 == z3.sat and r2 == z3.sat:
        return "neutral"
    raise Undecided("timeout")


def _linked(a, b, stated):
    """a and b connected by stated containment (either direction, transitively)."""
    adj = {}
    for u, v in stated:
        adj.setdefault(u, set()).add(v)
    def reach(x, y, seen=()):
        return x == y or any(reach(n, y, seen + (x,)) for n in adj.get(x, ()) if n not in seen)
    return reach(a, b) or reach(b, a)

