"""Dead reckoning for route and survey descriptions (Route->Survey).

A parse lists the description's events in its own terms (left / right turns and sides for route text, compass words
for survey text) and the question. The solver tracks the traveler's heading and position through the events, treats
unstated distances as positive unknowns (an answer is returned only if it holds for every assignment), and reads the
answer in the question's frame.

Parse = {"start": {"road": R, "facing": DIR | null}, "events": [EVENT], "question": QUESTION}; see
``enfor.tasks.route_survey`` for the event and question forms.
"""

import itertools
import math

from enfor.solvers import Undecided as _Undecided

COMPASS = {"north": 0, "northeast": 45, "east": 90, "southeast": 135, "south": 180, "southwest": 225, "west": 270,
           "northwest": 315}


EGO = {"front": 0, "ahead": 0, "right": 90, "back": 180, "behind": 180, "left": 270}


EIGHT = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]


GAPS = (1e-3, 1.0, 50.0)


TOL = 1e-9


class Undecided(_Undecided):
    pass


class Infeasible(Exception):
    pass


def vec(a):
    return (math.sin(math.radians(a)), math.cos(math.radians(a)))


def norm_dir(d):
    return str(d).lower().replace("-", "").replace(" ", "").replace("the", "") if d is not None else None


def absolute(d, h):
    """Compass angle of a direction word given heading h (ego words are relative to h)."""
    d = norm_dir(d)
    if d in COMPASS:
        return COMPASS[d]
    if d in EGO:
        return (h + EGO[d]) % 360
    raise Undecided(f"direction {d!r}")


class World:
    def __init__(self, rec, facing, gaps):
        self.gaps = iter(gaps)
        self.roads = {}   # name -> {"o": origin, "ax": axis angle, "pts": [points]}
        self.pts = {}     # name -> {"p": pos, "road": name, "side": angle}
        self.pos = (0.0, 0.0); self.h = facing; self.road = None; self.left_at = {}; self.hub = {}; self.loop = False; self.turned = 0; self.ambiguous = set(); self.bound = {}
        st = rec.get("start") or {}
        if st.get("facing"):
            self.h = absolute(st["facing"], 0)
        if self.h is None:
            self.h = 0
        if st.get("road"):
            self.on(st["road"])
        for e in rec.get("events") or []:
            self.event(e)

    def _find(self, x):
        for k in list(self.roads) + list(self.pts):
            if str(k).lower() == str(x).lower():
                return k
        return None

    def to_line(self, r):
        R = self.roads[r]; ax = vec(R["ax"]); v = vec(self.h)
        den = v[0] * (-ax[1]) + v[1] * ax[0]
        if abs(den) < TOL:
            raise Undecided("turn onto a parallel road")
        d = ((R["o"][0] - self.pos[0]) * (-ax[1]) + (R["o"][1] - self.pos[1]) * ax[0]) / den
        if d < -TOL:
            raise Infeasible("road behind")
        self.move(d)

    def on(self, r):
        self.road = r
        if r in self.roads and "far" in self.roads[r]:
            del self.roads[r]          # a boundary known by side only, now traversed
        if r in self.roads:
            R = self.roads[r]
            if abs(math.sin(math.radians(self.h - R["ax"]))) > 1e-6 or abs(_coords(self.pos, R["o"], R["ax"])[1]) > 1e-6:
                self.ambiguous.add(r)   # one name for two different roads
        R = self.roads.setdefault(r, {"o": self.pos, "ax": self.h, "pts": []})
        R["pts"].append(self.pos)

    def move(self, d):
        v = vec(self.h); self.pos = (self.pos[0] + d * v[0], self.pos[1] + d * v[1])
        if self.road:
            self.roads[self.road]["pts"].append(self.pos)

    def event(self, e):
        if "go" in e:
            self.move(next(self.gaps) if e["go"] is None else float(e["go"]))
        elif "close" in e:
            self.close(e["close"]); self.loop = True
        elif "end" in e:
            pass
        elif "see" in e:
            self.pts[e["see"]] = {"p": self.pos, "road": self.road, "side": absolute(e.get("side"), self.h)}
        elif "turn" in e and "onto" in e:
            g = next(self.gaps)
            tgt = self._find(e["onto"])
            if tgt is None or tgt not in self.roads or tgt == self.road or "far" in self.roads[tgt] or \
                    abs(math.sin(math.radians(self.roads[tgt]["ax"] - self.h))) < 1e-6:
                self.move(g)        # a new road (a name already used for another line is marked ambiguous in on())
            else:
                self.to_line(tgt)   # drive up to the known road (Infeasible if it is behind: skip this completion)
            if self.road:
                self.left_at[self.road] = self.pos
            h0 = self.h; self.h = absolute(e["turn"], self.h); self.on(e["onto"])
            self.turned += (self.h - h0 + 180) % 360 - 180
        elif "back_to" in e:
            r = e["back_to"]
            if r not in self.left_at:
                raise Undecided("back_to unknown road")
            self.pos = self.left_at[r]; self.h = absolute(e.get("turn"), (self.h + 180) % 360); self.on(r)
        elif "hub" in e:
            self.pos = (0.0, 0.0); self.hub = {"name": e["hub"], "roads": {}}
            h0 = self.h
            for r, d in (e.get("roads") or {}).items():
                self.hub["roads"][r] = absolute(d, h0)
        elif "onto" in e:
            r = e["onto"]
            if r not in self.hub.get("roads", {}):
                raise Undecided("onto unknown road")
            self.pos = (0.0, 0.0); self.h = self.hub["roads"][r]; self.on(r)
        elif "return_to" in e:
            self.pos = (0.0, 0.0); self.road = None
        elif "cross" in e:
            a = absolute(e.get("side"), self.h)
            self.roads.setdefault(e["cross"], {"o": self.pos, "ax": a, "pts": [self.pos, _add(self.pos, vec(a), 1e3)]})
            self.bound.setdefault(e["cross"], None)
            if self.road:
                self.bound.setdefault(self.road, None)
        elif "parallel" in e:
            a = absolute(e.get("side"), self.h); far = _add(self.pos, vec(a), 1e3)
            self.roads.setdefault(e["parallel"], {"o": far, "ax": self.h, "pts": [_add(far, vec(self.h), -1e4), _add(far, vec(self.h), 1e4)], "far": 1})
            self.bound.setdefault(e["parallel"], None)
            if self.road:
                self.bound.setdefault(self.road, None)
        elif "boundary" in e:
            self.bound[e["boundary"]] = absolute(e.get("side"), self.h)
            a = self.bound[e["boundary"]]; far = vec(a)
            o = (1e3 * far[0], 1e3 * far[1])
            self.roads.setdefault(e["boundary"], {"o": o, "ax": a + 90, "pts": [_add(o, vec(a + 90), -1e4),
                                                                                _add(o, vec(a + 90), 1e4)], "far": 1})

    def close(self, r):
        """Continue until the start road r, turn onto it and arrive at the start point (loop closure)."""
        r = self.name(r)
        R = self.roads[r]; ax = vec(R["ax"]); v = vec(self.h)
        den = v[0] * (-ax[1]) + v[1] * ax[0]
        if abs(den) < TOL:
            raise Undecided("close: parallel")
        d = ((R["o"][0] - self.pos[0]) * (-ax[1]) + (R["o"][1] - self.pos[1]) * ax[0]) / den
        if d < -TOL:
            raise Infeasible("close: start road behind")
        self.move(d); corner = self.pos
        if _coords(R["o"], corner, R["ax"])[0] < -TOL:
            raise Infeasible("close: start point not ahead on the start road")
        self.h = R["ax"]; self.road = r; R["pts"].append(corner)

    # geometry ------------------------------------------------------------------------------------------------
    def kind(self, x):
        if x in self.pts:
            return "point"
        if x in self.roads:
            return "road"
        for k in list(self.pts) + list(self.roads):
            if str(k).lower() == str(x).lower():
                return self.kind(k)
        raise Undecided(f"unknown entity {x!r}")

    def name(self, x):
        for k in list(self.pts) + list(self.roads):
            if str(k).lower() == str(x).lower():
                if k in self.ambiguous:
                    raise Undecided("ambiguous road name")
                return k
        raise Undecided(f"unknown entity {x!r}")

    def samples(self, x):
        x = self.name(x)
        return [self.pts[x]["p"]] if x in self.pts else self.roads[x]["pts"]


def _add(p, v, s):
    return (p[0] + s * v[0], p[1] + s * v[1])


def _coords(p, o, a):
    """(axial, lateral-right) coordinates of p in the frame at o heading a."""
    ax = vec(a); rt = vec(a + 90); d = (p[0] - o[0], p[1] - o[1])
    return d[0] * ax[0] + d[1] * ax[1], d[0] * rt[0] + d[1] * rt[1]


def _sign_all(vals):
    nz = [v for v in vals if abs(v) > TOL]
    if nz and all(v > 0 for v in nz):
        return 1
    if nz and all(v < 0 for v in nz):
        return -1
    return 0


def _compass(vx, vy):
    if abs(vx) < TOL and abs(vy) < TOL:
        raise Undecided("same place")
    a = math.degrees(math.atan2(vx, vy)) % 360
    return EIGHT[int(round(a / 45)) % 8]


def _opp(d):
    return EIGHT[(EIGHT.index(d) + 4) % 8]


def _snap4(v):
    a = math.degrees(math.atan2(v[0], v[1])) % 360
    return EIGHT[(int(round(a / 90)) % 4) * 2]


def road_pos(W, r):
    """Where a road lies in the area: its direction from the hub (star), its side of the loop (loop) or of the
    bounded area (bounded)."""
    if W.bound:
        if W.bound.get(r) is not None:
            return EIGHT[int(round(W.bound[r] / 45)) % 8]
        B = [W.roads[b] for b in W.bound if b in W.roads]
        c = (sum(R["o"][0] for R in B) / len(B), sum(R["o"][1] for R in B) / len(B))
        R = W.roads[r]; side = _coords(R["o"], c, R["ax"])[1]
        return _snap4(vec(R["ax"] + (90 if side > 0 else -90)))
    if W.hub:
        return EIGHT[int(round(W.hub["roads"][r] / 45)) % 8]
    if not W.loop:
        raise Undecided("no area structure")
    # the loop's interior is on the side the tour turns to; a road's side of the area is the other side
    out = W.roads[r]["ax"] + (90 if W.turned < 0 else -90)
    return _snap4(vec(out))


def survey_dir(W, target, frm):
    """Direction of target from frm. Road conventions depend on the area's structure (benchmark conventions):
    star / loop: a road's direction is where it lies in the area (from the hub / its side of the loop);
    otherwise roads are geometric (a side road lies to its side; a through road lies behind a side road's start)."""
    target, frm = W.name(target), W.name(frm)
    kt, kf = W.kind(target), W.kind(frm)
    on = kt == "point" and kf == "road" and W.pts[target]["road"] == frm
    if on:
        return EIGHT[int(round(W.pts[target]["side"] / 45)) % 8]
    if W.bound:
        if kt == "road" and target in W.bound:
            return road_pos(W, target)
        if kf == "road" and frm in W.bound:
            return _opp(road_pos(W, frm))
    structured = bool(W.hub) or W.loop
    if structured and kt == "road" and kf == "road":
        return road_pos(W, target)
    if structured and kt == "point" and kf == "road":
        return road_pos(W, W.pts[target]["road"]) if W.hub else _opp(road_pos(W, frm))
    if W.loop and kt == "point" and kf == "point" and W.pts[target]["road"] != W.pts[frm]["road"]:
        a, b = vec(COMPASS[road_pos(W, W.pts[target]["road"])]), vec(COMPASS[_opp(road_pos(W, W.pts[frm]["road"]))])
        return _compass(a[0] + b[0], a[1] + b[1])
    if kt == "point" and kf == "point":
        p, q = W.samples(target)[0], W.samples(frm)[0]
        dx = 0 if abs(p[0] - q[0]) < TOL else p[0] - q[0]; dy = 0 if abs(p[1] - q[1]) < TOL else p[1] - q[1]
        return _compass(math.copysign(1, dx) if dx else 0, math.copysign(1, dy) if dy else 0)
    if kf == "road":
        R = W.roads[frm]; ax = R["ax"]
        span = [_coords(x, R["o"], ax)[0] for x in R["pts"]]; lo, hi = min(span), max(span)
        cs = [_coords(x, R["o"], ax) for x in W.samples(target)]
        lat = _sign_all([c[1] for c in cs]); axs = 0
        if not lat:
            axs = 1 if all(c[0] >= hi - TOL for c in cs) else -1 if all(c[0] <= lo + TOL for c in cs) else 0
        if not lat and not axs:
            raise Undecided("road relation undetermined")
        v = vec(ax + 90) if lat > 0 else vec(ax - 90) if lat < 0 else vec(ax if axs > 0 else ax + 180)
        return _compass(v[0], v[1])
    return _opp(survey_dir(W, frm, target))


def route_where(W, q):
    """Answer a route question: left / right / front / back."""
    tgt = W.name(q["target"])
    if q.get("from") is not None:
        A, B = W.name(q["from"]), W.name(q["to"])
        pa, pb = W.samples(A)[0], W.samples(B)[0]
        h = math.degrees(math.atan2(pb[0] - pa[0], pb[1] - pa[1])) % 360; o = pa
        a, b = 0.0, math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        road = W.pts[A]["road"]
    else:
        road = W.name(q["along"]); R = W.roads[road]; ref = W.name(q["ref"]); want = norm_dir(q.get("ref_pos"))
        h = None
        for s in (R["ax"], (R["ax"] + 180) % 360):
            if ref_position(W, ref, road, s) == want:
                if h is not None:
                    raise Undecided("ambiguous travel direction")
                h = s
        if h is None:
            raise Undecided("no travel direction fits")
        o = R["o"]; span = [_coords(p, o, h)[0] for p in R["pts"]]; a, b = min(span), max(span)
    side = lateral(W, tgt, road, h)
    if q.get("ask") == "side":
        if not side:
            raise Undecided("no side")
        return side
    if W.kind(tgt) == "point" and W.pts[tgt]["road"] == road and q.get("from") is None:
        return side
    if W.kind(tgt) == "road" and _parallel_offset(W, tgt, road):
        if not side:
            raise Undecided("no side")
        return side
    us = [_coords(p, o, h)[0] for p in W.samples(tgt)]
    if max(us) <= a + TOL:
        return "back"
    if min(us) >= b - TOL:
        return "front"
    if not side:
        raise Undecided("no side")
    return side


def lateral(W, x, road, h):
    if W.kind(x) == "point" and W.pts[x]["road"] == road:
        d = (W.pts[x]["side"] - h) % 360
        return "right" if abs(d - 90) < 1 else "left" if abs(d - 270) < 1 else None
    R = W.roads[road]
    s = _sign_all([_coords(p, R["o"], h)[1] for p in W.samples(x)])
    return "right" if s > 0 else "left" if s < 0 else None


def _parallel_offset(W, x, road):
    """Road x runs parallel to road, on a different line (its side is then its lateral side)."""
    X, R = W.roads[x], W.roads[road]
    return abs(math.sin(math.radians(X["ax"] - R["ax"]))) < 1e-6 and abs(_coords(X["o"], R["o"], R["ax"])[1]) > 1e-6


def ref_position(W, ref, road, h):
    """left / right / front / back of a reference while travelling along road with heading h (a road that lies at or
    beyond an end of the travelled road is ahead / behind; otherwise its side)."""
    if W.kind(ref) == "point" and W.pts[ref]["road"] == road:
        return lateral(W, ref, road, h)
    if W.kind(ref) == "road" and _parallel_offset(W, ref, road):
        return lateral(W, ref, road, h)
    R = W.roads[road]; o = R["o"]; span = [_coords(p, o, h)[0] for p in R["pts"]]; a, b = min(span), max(span)
    us = [_coords(p, o, h)[0] for p in W.samples(ref)]
    if max(us) <= a + TOL:
        return "back"
    if min(us) >= b - TOL:
        return "front"
    return lateral(W, ref, road, h)


def answer(rec, facing=None):
    """Answer that holds for every tested value of the unstated distances (else Undecided)."""
    n_gaps = sum(1 for e in rec.get("events") or [] if ("turn" in e and "onto" in e) or ("go" in e and e["go"] is None))
    combos = [tuple([g] * n_gaps) for g in GAPS]
    # extreme rays of the positive orthant: each unknown alone large / alone small (every coordinate is affine in the
    # unknowns, so its sign is constant over the orthant iff it is constant at these points)
    lo, hi = GAPS[0], GAPS[-1]
    for i in range(n_gaps):
        combos.append(tuple(hi if j == i else lo for j in range(n_gaps)))
        combos.append(tuple(lo if j == i else hi for j in range(n_gaps)))
    combos += [c for c in itertools.islice(itertools.product(GAPS, repeat=n_gaps), 0, 27) if c not in combos]
    fac = COMPASS.get(norm_dir(facing)) if facing else None
    outs = set()
    for c in combos:
        try:
            W = World(rec, fac, c)
        except Infeasible:
            continue
        q = rec["question"]
        outs.add(survey_dir(W, q["target"], q["from"]) if q.get("ask") == "direction" else route_where(W, q))
        if len(outs) > 1:
            raise Undecided(f"depends on unstated distances: {sorted(outs)}")
    if not outs:
        raise Undecided("no feasible completion")
    return outs.pop()

