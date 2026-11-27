"""Collision filter (cell 23 by default, `use_cell("robot20")` for cell 20): decides whether one commanded move may jam the gripper.

    verdict = check(pose0, pose1, states)   # poses [x, y, z, rotation / 180, gripper], states from SceneTracker

Blocked (these jam and break the gripper):
  - pushing an object, or a chain of touching objects, that the frame stops from moving;
  - a finger meeting the frame itself;
  - lowering a finger onto an object or the frame;
  - opening the gripper into the frame, or into an object that the frame stops (cell 20; closing is a grasp, and
    an object clamped between the tips may be lowered and let go);
  - any move but straight up while an object is unaccounted for.
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
HELD_BELOW_MM = 0.6  # a held object's bottom below the tips' (grasped from the glass: the tips are 0.6 mm up at z 0)

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
STACK_GROW_MM = 10.0  # an object with a cube stacked on it grows by this much all round (one cube: see _obstacles)
NOTCH_STACK_TOP_MM = 18.5  # a cube in the slot's notch: on its 5 mm floor, tilted at most until it wedges between
                           # the 13.5 mm walls (10 (cos t + sin t) <= 13.5), so its top is at most 5 + 13.5 mm
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
    # Wider (2026-11-27, in the air, top view, as ratios to 0.825): 28.7 mm at 0.9, 29.7 mm at 1.0; the linkage flattens
    # (a straight line through the first two would say 32 mm). np.interp clamps past the last point: keep it measured.
    GRIP_TO_SEP_MM=((0.1, 9.0), (0.825, 27.5), (0.9, 28.7), (1.0, 29.7)),
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
    held: list[str] = field(default_factory=list)  # objects in the grasp after this move: pass them to the next check


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


def _obstacles(states, held=()) -> tuple[dict, dict]:
    """Rasterised footprints and top heights (mm) of everything a finger can meet, by object name. Held objects are
    none: they move with the fingers (lifted, the tracker sees them "stacked" on whatever they are carried over;
    prim_12, 2026-11-26, that grew the slot under a carried cube and blocked the carry and the release).

    ponytail: a held object's own outline is taken as inside the fingers' (true for a cube or the slot held low);
    the pan held by its handle sticks out up to 16 mm, unchecked."""
    tops = {n: s.height_mm for n, s in states.items() if n not in held and s.status == "on_glass" and s.footprint_xy is not None}
    masks = {n: _raster([states[n].footprint_xy]) for n in tops}
    for n, s in states.items():
        if n in held or (s.status == "stacked" and s.on in held):  # in the grasp, or on what is: moves with it
            continue
        if s.status == "stacked" and s.on in tops:
            # "stacked on X" only means off the glass and touching X in the top view: the cube may also lean
            # beside X (on the rope, say). So X gets the cube's height and grows by a cube all round; not for a cube
            # the top view puts inside the slot's outline (in its notch it cannot lean beside the slot).
            if getattr(s, "in_notch", False):
                tops[s.on] = max(tops[s.on], NOTCH_STACK_TOP_MM)
            else:
                tops[s.on] += s.height_mm
                masks[s.on] = _grow(masks[s.on], STACK_GROW_MM)
        elif s.status in ("stacked", "off_glass") and getattr(s, "inferred_xy", None) is not None:
            # off the glass and on nothing we see from below (on the frame, most likely): an obstacle of its own
            # where the top view puts it. ponytail: that position is rough, hence the generous square, and the
            # height assumes it rests on something as tall as the frame
            r = (STACK_GROW_MM + 5) * MM * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
            masks[n], tops[n] = _raster([np.array(s.inferred_xy) + r]), FRAME_MM + s.height_mm
    return masks, tops


def _chain(first: list[str], masks: dict) -> list[str]:
    """`first` plus every object that touches one of them, and so on: they move together."""
    chain = list(first)
    for n in chain:
        chain += [o for o, m in masks.items() if o not in chain and (m & _grow(masks[n], CONTACT_MM)).any()]
    return chain


def _jammed(chain: list[str], masks: dict, d_cells, extra_mm: float = 0.0) -> str | None:
    """The first object of `chain` that the frame would stop if all were shifted by d_cells (grid cells, x, y).

    It ends up deeper in the frame's margin than it was: sliding along the frame is not a jam.
    """
    shift = np.float32([[1, 0, d_cells[1]], [0, 1, d_cells[0]]])
    for n in chain:
        moved = cv2.warpAffine(masks[n], shift, (_N, _N), flags=cv2.INTER_NEAREST)
        before, after = (int((_grow(m, JAM_MM + a) & _FRAME).sum()) for m, a in ((masks[n], 0.0), (moved, extra_mm)))
        if after > before + 10:  # 10 cells of raster noise
            return n
    return None


def _clamped(pose, masks: dict, tops: dict) -> list[str]:
    """Objects the tips clamp at `pose`: the gripper is commanded narrower than the object, so BOTH tips as drawn
    overlap it (the real ones are stopped by it), and the tips are clearly below its top. Closed tips hovering
    just above a cube overlap it in the same way: hence the height condition, and hence `held` is only set by a
    closing move (see check)."""
    tip, (a, b) = tip_mm(pose[2]), (_raster([f]) for f in finger_polys(pose))
    return [n for n, m in masks.items() if tops[n] > tip + CLEAR_MM and (m & a).any() and (m & b).any()]


def _hull(a, b) -> np.ndarray:
    return cv2.convexHull(np.float32(np.vstack([a, b]))).reshape(-1, 2)


def check(pose0, pose1, states, held=()) -> Verdict:
    """May the arm go from pose0 to pose1? `states` is SceneTracker's {name: ObjectState}.

    `held`: the objects in the grasp, which are no obstacle to lowering or to letting go. The caller keeps them:
    pass the previous sent move's `verdict.held` (a closing move adds what it clamps, an opening move empties it).
    Without it a grasped object can be carried but not put down again.

    The move is taken as the env sends it: x, y first, then z, then rotation, then the gripper.
    """
    pose0, pose1 = np.asarray(pose0, float), np.asarray(pose1, float)
    v = Verdict(False, "", fingers=finger_polys(pose1))
    missing = [n for n, s in states.items() if s.status == "unaccounted" and n not in held]
    up = pose1[2] > pose0[2] and np.allclose(np.delete(pose1, 2), np.delete(pose0, 2))
    if missing and not up:  # straight up meets nothing; staying down is worse (play_1 episode 4: the arm stayed at the slot)
        v.reason = f"blocked: {', '.join(missing)} unaccounted for"
        return v
    held = [n for n in held if n in states]
    masks, tops = _obstacles(states, held)

    # the fingers along the move: x, y at the old rotation, then the turn at the new x, y; the gripper stays as it was
    end = pose1.copy(); end[4] = pose0[4]
    mid = pose0.copy(); mid[:2] = end[:2]
    path = [pose0 + (mid - pose0) * t for t in np.linspace(0, 1, SAMPLES)]
    path += [mid + (end - mid) * t for t in np.linspace(0, 1, SAMPLES)[1:]] if end[3] != pose0[3] else []
    swept = _raster([_hull(a, b) for p, q in zip(path, path[1:] + path[-1:]) for a, b in zip(finger_polys(p), finger_polys(q))])
    reach = _grow(swept, CONTACT_MM)
    if "rope" in masks:
        masks["rope"] &= _grow(swept, ROPE_LOCAL_MM)

    # lowering: nothing may be under a finger that ends below its top
    tip0, tip1 = tip_mm(pose0[2]), tip_mm(pose1[2])
    under = _grow(_raster(finger_polys(end)), LOWER_MM)
    if tip1 < tip0:
        hit = [n for n, m in masks.items() if n not in held and tops[n] > tip1 - CLEAR_MM and (m & under).any()]
        if FRAME_MM > tip1 - CLEAR_MM and (_FRAME & under).any():
            hit.append("the frame")
        if held:  # what is held fills the space between the tips, its bottom HELD_BELOW_MM below theirs
            between = _grow(_raster([_hull(*finger_polys(end))]), LOWER_MM)
            bottom = tip1 - HELD_BELOW_MM
            hit += [n for n, m in masks.items() if n not in hit and tops[n] > bottom - CLEAR_MM and (m & between).any()]
            if "the frame" not in hit and FRAME_MM > bottom - CLEAR_MM and (_FRAME & between).any():
                hit.append("the frame")
        if hit:
            v.reason = f"blocked: lowering {', '.join(held) + ' ' if held else ''}onto {', '.join(hit)}"
            return v

    # sideways: the lowest the tips get decides what they can touch
    tip = min(tip0, tip1)
    if tip - CLEAR_MM < FRAME_MM and (_FRAME & reach).any():
        v.reason = "blocked: finger against the frame"
        return v
    chain = _chain([n for n, m in masks.items() if tops[n] > tip - CLEAR_MM and (m & reach).any()], masks)
    # would the frame stop any of them? Shift each by the travel (a turn pushes any way: grow by the arc instead)
    arc = abs(end[3] - pose0[3]) * np.pi * (sep_mm(pose0[4]) / 2 + FINGER_MM[0])
    n = _jammed(chain, masks, (end[:2] - pose0[:2]) * _PX, arc)
    if n:
        v.reason = f"blocked: pushes {', '.join(chain)} with {n} against the frame"
        return v

    # the gripper, last and at the final pose. Closing on an object is a grasp and allowed. Opening pushes what the
    # tips meet outwards along the gripper axis: blocked if that is the frame, or an object the frame would stop.
    half = (sep_mm(pose1[4]) - sep_mm(pose0[4])) / 2 * MM
    if half:
        axis = np.array([np.cos(pose1[3] * np.pi), np.sin(pose1[3] * np.pi)])
        for side, a, b in zip((-1, 1), finger_polys(end), v.fingers):
            path_g = _grow(_raster([_hull(a, b)]), CONTACT_MM)
            if half > 0 and tip1 - CLEAR_MM < FRAME_MM and (_FRAME & path_g).any():
                v.reason = "blocked: finger opening against the frame"
                return v
            met = [n for n, m in masks.items() if n not in held and tops[n] > tip1 - CLEAR_MM and (m & path_g).any()]
            if half > 0:
                met = _chain(met, masks)
                n = _jammed(met, masks, side * half * axis * _PX)
                if n:
                    v.reason = f"blocked: opening pushes {', '.join(met)} with {n} against the frame"
                    return v
            chain += [n for n in met if n not in chain]
    v.allowed, v.pushed = True, chain
    v.held = [] if half > 0 else held + [n for n in _clamped(pose1, masks, tops) if n not in held] if half < 0 else held
    v.reason = f"pushes {', '.join(chain)}" if chain else "free"
    return v
