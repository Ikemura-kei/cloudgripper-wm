"""Collision filter (cell 23 by default, `use_cell("robot20")` for cell 20): decides whether one commanded move may jam the gripper.

    verdict = check(pose0, pose1, states)   # poses [x, y, z, rotation / 180, gripper], states from SceneTracker

Blocked (these jam and break the gripper):
  - pushing an object, or a chain of touching objects, that the frame stops from moving;
  - a finger meeting the frame itself;
  - lowering a finger onto an object or the frame;
  - any move while an object is unaccounted for.
Allowed: pushing a free object (reported in `verdict.pushed`), and everything that touches nothing.

Pure geometry on a 0.5 mm grid in robot x, y; no GPU, no camera. Use the COMMANDED pose: the robot's reported
x, y trail the arm by seconds to minutes.

The numbers below are for cell 23 with its gripper stuck open (2026-11-24); CELL20 further down holds cell 20's.
All were measured in one or two base camera images. Check them with the overlay of scripts/debug/perception_replay.py before trusting a new setup.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MM = 1 / 140.0  # robot units per mm (the 0 to 1 range spans 140 mm)

# Glass interior in robot x, y: SceneTracker's CELLS["robot23"]["glass"] pixel rectangle through the cell's
# camera-to-robot fit. Outside it is the frame.
GLASS = (0.100, 0.111, 0.892, 0.873)  # x0, y0, x1, y1
# Frame height above the glass, from its CAD model: the parent repo's assets/"Part Studio 1 - Part 1.stl" (one half
# of the frame) is 6.25 mm thick; Kei read 6 mm in CAD (2026-11-24). That is level with the tips' lowest point
# (6 mm), so with CLEAR_MM the fingers count as touching the frame below about z 0.12 and pass over it above.
# ponytail: assumes the frame lies flat on the glass with nothing under it
FRAME_MM = 6.25

# Finger tips seen from below, gripper stuck open. Measured 2026-11-24 at rotation 9 deg and 2026-11-23 at 0 deg.
# The gripper is off its servo and stays where it is put by hand: open on the morning of 2026-11-24, closed by Kei
# that afternoon (tips 7 mm apart). The default stays the open value because it is the wider, more cautious one;
# pass the real spacing (--finger-sep-mm 7 in the debug scripts) while it is closed.
FINGER_SEP_MM = 23.0   # centre to centre, along the gripper axis (robot x at rotation 0)
FINGER_MM = (5.5, 9.5)  # one tip: along the axis, across it
# Midpoint of the tips minus the commanded x, y. In the base image the tips sit 8 to 11 px (x) and 0 to -19 px (y)
# from the commanded pose's pixel; the y part grows with the distance from the image centre, so it is read as
# parallax of tips 13 mm above the glass, not as a real offset, and left out.
# ponytail: three images, one rotation, and CONTACT_MM no longer covers the doubt. If contact comes early from one
# side and late from the opposite one, correct it here. If the offset turns with the gripper, rotate it here.
TOOL_OFFSET = (0.020, 0.0)

# Tip height above the glass against the height command, measured by Kei with a ruler on 2026-11-24 (gripper
# closed by hand; 0.5 measured twice, once arriving from above and once from below). The tips never reach the glass.
# TODO after the gripper repair (Kei, 2026-11-24): measure these three heights again, and FINGER_SEP_MM, FINGER_MM
# and TOOL_OFFSET with them. Until then these numbers stand.
# ponytail: measured with the fingers closed; if the open fingers' tips sit higher or lower, they are off
Z_TO_TIP_MM = ((0.0, 6.0), (0.5, 15.0), (1.0, 22.0))  # interpolated in between
CLEAR_MM = 2.0  # a tip less than this above an object's top counts as touching it (ruler and height error)

# Sideways: a finger this close to an object counts as touching it. Tuned by eye in the teleop (Kei, 2026-11-24,
# gripper closed): at 4 the filter reported contact two steps before the real one, at 1.5 it was "kind of neat so far".
CONTACT_MM = 1.5
# Lowering: an object this close to a finger, seen from above, counts as under it. It must stay small or the open
# fingers cannot go down around a cube: the gap is 17.5 mm, a cube 10 mm wide and 14.1 mm across its diagonal, so
# 1.5 mm lets a cube through at any angle if it is centred within 0.2 mm, and a cube square to the fingers within
# 2.2 mm. ponytail: that trusts FINGER_SEP_MM, FINGER_MM and TOOL_OFFSET to about 1 mm, which three images do not
# give; measure them on the repaired gripper before the first real lowering around a cube.
LOWER_MM = 1.5
JAM_MM = 2.0      # an object this close to the frame, after the push, counts as stopped by it (5 blocked too early; same tuning)
ROPE_LOCAL_MM = 25.0  # only this much rope around the fingers is treated as pushed: the rest just bends
SAMPLES = 8       # poses rasterised along one move

_PX = 280  # grid cells per robot unit (0.5 mm)
_LO, _N = -0.15, 364  # the grid covers -0.15 to 1.15


# Cell 20 (2026-11-25, working gripper). use_cell("robot20") puts these in place of the cell 23 values above.
CELL20 = dict(
    # SceneTracker's CELLS["robot20"]["glass"] through cell 20's camera-to-robot fit.
    # ponytail: the frame's inner corners are rounded (about 8 mm); the rectangle lets a finger into them
    GLASS=(0.043, 0.052, 0.842, 0.845),
    # Tip spacing (centre to centre) against the gripper command, from the tips' outer edges in base pictures at
    # z 0 (outputs/cloudgripper/robot20/scene1/fing_*): 14.2 mm outer to outer closed at 0.1, 32.3 mm open at 0.825.
    GRIP_TO_SEP_MM=((0.1, 9.0), (0.825, 27.5)),
    # ponytail: tip and finger body are the same yellow, so the inner edges may include body: the boxes may be up
    # to about 1 mm too deep along the axis and too wide across it, which is the cautious side
    FINGER_MM=(5.0, 10.5),
    # Tips' midpoint minus the commanded x, y, read off the same pictures: (0.010, 0.022) open and (0.007, 0.010)
    # closed at rotation 0; about (0.004, 0.007) open and (0.004, -0.001) closed at 90 deg. So a fixed part (the
    # rotation axis is not at the commanded point) plus a part that turns with the gripper; these two fit the four
    # readings within 0.8 mm. ponytail: the turning part also grows a little as the gripper opens; left out.
    TOOL_OFFSET=(0.0125, 0.0075),
    TOOL_OFFSET_TURNING=(-0.0044, 0.0085),
    # Kei's ruler, gripper closed at 0.1: the tips all but reach the glass.
    Z_TO_TIP_MM=((0.0, 0.6), (0.5, 13.5), (1.0, 27.0)),
)
TOOL_OFFSET_TURNING = (0.0, 0.0)  # cell 23: not measured
GRIP_TO_SEP_MM = None  # cell 23: the gripper does not move, FINGER_SEP_MM holds


def use_cell(robot: str) -> None:
    """Switch the module to this cell's numbers (only robot20 differs from the defaults; call once, at start)."""
    global _FRAME
    if robot == "robot20":
        globals().update(CELL20)
        _FRAME = _frame()


def sep_mm(grip: float) -> float:
    return FINGER_SEP_MM if GRIP_TO_SEP_MM is None else float(np.interp(grip, *zip(*GRIP_TO_SEP_MM)))


@dataclass
class Verdict:
    allowed: bool
    reason: str  # "free", "pushes <names>", or why it is blocked
    pushed: list[str] = field(default_factory=list)  # objects the move touches, with those they touch in turn
    fingers: list[np.ndarray] = field(default_factory=list)  # two (4, 2) polygons at pose1, robot x, y (for drawing)


def tip_mm(z: float) -> float:
    return float(np.interp(z, *zip(*Z_TO_TIP_MM)))


def finger_polys(pose) -> list[np.ndarray]:
    """The two finger tips as (4, 2) polygons in robot x, y."""
    th = pose[3] * np.pi
    c = np.array(pose[:2], float) + TOOL_OFFSET + np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ TOOL_OFFSET_TURNING
    ax, across = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    a, b = FINGER_MM[0] / 2 * MM * ax, FINGER_MM[1] / 2 * MM * across
    return [c + s * sep_mm(pose[4]) / 2 * MM * ax + np.array([a + b, a - b, -a - b, -a + b]) for s in (-1, 1)]


def _raster(polys) -> np.ndarray:
    m = np.zeros((_N, _N), np.uint8)
    for p in polys:  # image row = x, column = y; any consistent choice works
        cv2.fillPoly(m, [np.round((np.asarray(p, float)[:, ::-1] - _LO) * _PX).astype(np.int32)], 1)
    return m


def _grow(m: np.ndarray, mm: float) -> np.ndarray:
    k = 2 * int(round(mm * MM * _PX)) + 1
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def _frame() -> np.ndarray:
    m = np.ones((_N, _N), np.uint8)
    x0, y0, x1, y1 = (int(round((v - _LO) * _PX)) for v in GLASS)
    m[x0:x1, y0:y1] = 0
    return m


_FRAME = _frame()


def check(pose0, pose1, states) -> Verdict:
    """May the arm go from pose0 to pose1? `states` is SceneTracker's {name: ObjectState}.

    The move is taken as the env sends it: x, y first, then z, then rotation.
    """
    pose0, pose1 = np.asarray(pose0, float), np.asarray(pose1, float)
    v = Verdict(False, "", fingers=finger_polys(pose1))
    missing = [n for n, s in states.items() if s.status == "unaccounted"]
    if missing:
        v.reason = f"blocked: {', '.join(missing)} unaccounted for"
        return v

    # obstacles: every object on the glass with what is stacked on it, and the frame
    tops = {n: s.height_mm for n, s in states.items() if s.status == "on_glass" and s.footprint_xy is not None}
    for s in states.values():
        if s.status == "stacked" and s.on in tops:
            tops[s.on] += s.height_mm
    # ponytail: a cube off the glass that rests on the rope beside its "on" object is not an obstacle of its own;
    # give it one from its top-view position before lowering is allowed
    masks = {n: _raster([states[n].footprint_xy]) for n in tops}

    # the fingers along the move: x, y at the old rotation, then the turn at the new x, y
    mid = pose0.copy(); mid[:2] = pose1[:2]
    path = [pose0 + (mid - pose0) * t for t in np.linspace(0, 1, SAMPLES)]
    path += [mid + (pose1 - mid) * t for t in np.linspace(0, 1, SAMPLES)[1:]] if pose1[3] != pose0[3] else []
    swept = _raster([cv2.convexHull(np.float32(np.vstack([a, b]))).reshape(-1, 2)  # hull of consecutive samples
                     for p, q in zip(path, path[1:] + path[-1:]) for a, b in zip(finger_polys(p), finger_polys(q))])
    reach = _grow(swept, CONTACT_MM)
    if "rope" in masks:
        masks["rope"] &= _grow(swept, ROPE_LOCAL_MM)

    # lowering: nothing may be under a finger that ends below its top
    tip0, tip1 = tip_mm(pose0[2]), tip_mm(pose1[2])
    under = _grow(_raster(v.fingers), LOWER_MM)
    if tip1 < tip0:
        hit = [n for n, m in masks.items() if tops[n] > tip1 - CLEAR_MM and (m & under).any()]
        if FRAME_MM > tip1 - CLEAR_MM and (_FRAME & under).any():
            hit.append("the frame")
        if hit:
            v.reason = f"blocked: lowering onto {', '.join(hit)}"
            return v

    # sideways: the lowest the tips get decides what they can touch
    tip = min(tip0, tip1)
    if tip - CLEAR_MM < FRAME_MM and (_FRAME & reach).any():
        v.reason = "blocked: finger against the frame"
        return v
    chain = [n for n, m in masks.items() if tops[n] > tip - CLEAR_MM and (m & reach).any()]
    for n in chain:  # objects touching a pushed object are pushed too
        chain += [o for o, m in masks.items() if o not in chain and (m & _grow(masks[n], CONTACT_MM)).any()]
    if not chain:
        v.allowed, v.reason = True, "free"
        return v
    v.pushed = chain
    # would the frame stop any of them? Shift each by the travel (a turn pushes any way: grow by the arc instead)
    # and see whether it ends up deeper in the frame's margin than it was: sliding along the frame is not a jam.
    d = (pose1[:2] - pose0[:2]) * _PX
    arc = abs(pose1[3] - pose0[3]) * np.pi * (sep_mm(pose1[4]) / 2 + FINGER_MM[0])
    shift = np.float32([[1, 0, d[1]], [0, 1, d[0]]])
    for n in chain:
        moved = cv2.warpAffine(masks[n], shift, (_N, _N), flags=cv2.INTER_NEAREST)
        before, after = (int((_grow(m, JAM_MM + a) & _FRAME).sum()) for m, a in ((masks[n], 0.0), (moved, arc)))
        if after > before + 10:  # 10 cells of raster noise
            v.reason = f"blocked: pushes {', '.join(chain)} with {n} against the frame"
            return v
    v.allowed, v.reason = True, f"pushes {', '.join(chain)}"
    return v
