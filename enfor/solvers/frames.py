"""Frame-of-reference solver for a single spatial relation (FoREST).

Geometry, seen from above: the camera sits at the origin and looks along +y; x points to the camera's right.
- A relatum's facing is given relative to the camera: toward the camera (-y), away (+y), left (-x), right (+x).
- Relative frame (camera's perspective): left -x, right +x, front -y (between camera and relatum), back +y.
- Intrinsic frame (relatum's perspective, facing f): front f, back -f, left = f rotated 90 degrees counter-clockwise,
  right = -left.
A statement places the locatum at a unit offset from the relatum in its frame; a question reads that offset in the
frame it names (camera, or the relatum's own).
"""


FACING = {"toward": (0, -1), "away": (0, 1), "left": (-1, 0), "right": (1, 0)}


REL_CAMERA = {"left": (-1, 0), "right": (1, 0), "front": (0, -1), "back": (0, 1)}


def rot90(v):
    return (-v[1], v[0])


def intrinsic_axes(f):
    left = rot90(f)
    return {"front": f, "back": (-f[0], -f[1]), "left": left, "right": (-left[0], -left[1])}


def place(relation, frame, facing):
    """Offset of the locatum from the relatum, in camera coordinates. frame: 'relative' | 'intrinsic'."""
    if frame == "relative":
        return REL_CAMERA[relation]
    if facing is None:
        raise ValueError("intrinsic frame needs the relatum's facing")
    return intrinsic_axes(FACING[facing])[relation]


def read(offset, view, facing):
    """Name the offset in the requested view: 'camera' or 'relatum'."""
    axes = REL_CAMERA if view == "camera" else intrinsic_axes(FACING[facing])
    for name, v in axes.items():
        if v == offset:
            return name
    raise ValueError("offset not axis-aligned")


def solve(relation, frame, facing, view):
    return read(place(relation, frame, facing), view, facing)

