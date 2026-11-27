"""Self-check of a play cell before and during data collection (Kei, 2026-11-27). MOVES THE ROBOT.

The arm visits a fixed list of poses (`_plan`) through the filtered env like every other move, never below height
LOW_Z (Kei). At each pose it takes both pictures and the robot's reported state, and compares them with a reference
recorded on the healthy robot at the same pose:
- every axis got there and the gripper looks as it did: in the top view, the yellow block (the gripper's fixed
  anchor; it moves with x, y and height, not with the rotation) and the claw under it (the finger linkages; the
  cable's thin yellow stripe is opened away); in the base view, all the yellow. Each mask must lie within NEAR_PX
  of the reference's (`_near`: closing twice leaves the fingers a few px apart);
- the gripper hangs straight: the angle of the block's bottom edge matches the reference within TILT_DEG (+2.3 deg
  at x 0.5, y 0.5, top height on 2026-11-26, open, half and closed alike; Kei: a broken one tilts about 30 deg);
- the base camera was not moved: the blue frame lies within FRAME_PX of the reference's (a moved top camera shows
  as the block off at every pose);
- the cameras deliver: no picture is the same array as the previous pose's (the env reuses the last frame when a
  fetch fails, which looks like an arm that did not move);
- the robot reports the commanded pose (catches an ignored command; a slipped belt shows only in the pictures).
Motor currents are kept in the reference's poses.json with the rest of the state, not judged (Kei: never used).

Measured on the 2026-11-26 pictures (same pose, top height, 6 px slack): claw same state 0.98 to 1.0, half-open
against closed 0.81, open against closed 0.19; base frame 0.986, shifted by 3 px 0.89, by 5 px 0.70.
ponytail: limits from those five pictures; check them against the first healthy runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from cloudgripper_wm import collision_filter as cf

C, R = (0.5, 0.5), 1.0  # reference pose: middle of the arm's range, rotation 180 deg (fingers side-on to the top camera)
HALF = 0.45            # half-open gripper command
LOW_Z = 0.35           # lowest height in the check (Kei: a wrongly free-looking spot is less of a risk up there)
LOW_SPOTS = ((0.5, 0.5), (0.3, 0.3), (0.7, 0.7))  # where the height is checked: the first one the filter allows
YELLOW, BLUE = ((18, 120, 120), (35, 255, 255)), ((100, 150, 50), (130, 255, 255))  # HSV ranges, cell 20
NEAR_PX, NEAR_MIN = 6, 0.92
FRAME_PX, FRAME_MIN = 2, 0.95
TILT_DEG = 5.0
POSE_TOL = 0.01        # reported against commanded, in the env's units (rotation 0.01 = 1.8 deg)


def _mask(img, hsv_range) -> np.ndarray:
    return cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), *hsv_range) > 0


def _pieces(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    return (None, None) if n < 2 else (lab == 1 + st[1:, 4].argmax(), st[1 + st[1:, 4].argmax()])


def _split(top) -> tuple[np.ndarray, np.ndarray]:
    """Top view: the gripper's yellow block (the largest yellow piece) and the claw under it."""
    y = _mask(top, YELLOW)
    blk, st = _pieces(y)
    if blk is None:
        return y, y
    x, w = st[0], st[2]
    claw = cv2.morphologyEx((y & ~blk).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)) > 0
    claw[:np.where(blk.any(1))[0].max()] = False
    claw[:, :max(0, x - w // 3)] = claw[:, x + w + w // 3:] = False
    return blk, claw


def _tilt(blk) -> float:
    """Angle (deg) of the block's bottom edge in the image; nan without a block."""
    xs = np.where(blk.any(0))[0]
    xs = xs[(xs > xs.min() + 15) & (xs < xs.max() - 15)] if len(xs) else xs  # without the rounded corners
    if len(xs) < 20:
        return float("nan")
    ys = blk.shape[0] - 1 - np.argmax(blk[::-1][:, xs], axis=0)  # lowest block pixel in each column
    return float(np.degrees(np.arctan(np.polyfit(xs, ys, 1)[0])))


def _near(a, b, px: int) -> float:
    """The smaller of: the share of a within px of b, and of b within px of a (1 = same shape, small shifts forgiven)."""
    if not a.any() and not b.any():
        return 1.0
    k = np.ones((2 * px + 1,) * 2, np.uint8)
    da, db = (cv2.dilate(m.astype(np.uint8), k) > 0 for m in (a, b))
    return float(min((a & db).sum() / max(a.sum(), 1), (b & da).sum() / max(b.sum(), 1)))


def _measure(top, base) -> dict:
    blk, claw = _split(top)
    frame = _pieces(_mask(base, BLUE))[0]
    return dict(block=blk, claw=claw, base=_mask(base, YELLOW), frame=np.zeros(base.shape[:2], bool) if frame is None else frame,
                tilt=_tilt(blk))


def _plan(prim, spots) -> list[tuple]:
    """(label, x, y, z, rot, grip) for each pose: every axis both ways at the top height, then down at each spot."""
    o, c, (x, y), up = prim.open, prim.closed, C, 1.0
    plan = [("open", x, y, up, R, o), ("closed", x, y, up, R, c), ("half", x, y, up, R, HALF),
            ("x+", x + .1, y, up, R, o), ("x-", x - .1, y, up, R, o), ("y+", x, y + .1, up, R, o), ("y-", x, y - .1, up, R, o),
            ("rot135", x, y, up, R - .25, o), ("rot90", x, y, up, R - .5, o), ("open_again", x, y, up, R, o)]
    for i in spots:
        sx, sy = LOW_SPOTS[i]
        plan += [(f"s{i}_over", sx, sy, up, R, o), (f"s{i}_low_open", sx, sy, LOW_Z, R, o),
                 (f"s{i}_low_closed", sx, sy, LOW_Z, R, c), (f"s{i}_up", sx, sy, up, R, c)]
    return plan


def _free(prim, i: int) -> bool:
    sx, sy = LOW_SPOTS[i]
    over, low = [sx, sy, 1.0, R, prim.open], [sx, sy, LOW_Z, R, prim.open]
    return cf.check(over, low, prim.env.states).allowed and cf.check(low, [*low[:4], prim.closed], prim.env.states).allowed


def _overlay(img, cur, ref) -> np.ndarray:
    """Reference only magenta, this check only cyan, both white."""
    o = img // 2
    o[ref & ~cur], o[cur & ~ref], o[cur & ref] = (255, 0, 255), (255, 255, 0), (255, 255, 255)
    return o


def self_check(prim, ref, out=None, record: bool = False) -> tuple[str, list[str]]:
    """Run the check with `prim` (Primitives on a FilteredCloudGripperWrapper after reset). Returns the status,
    "pass" | "fail" | "blocked" (the filter stopped a move: the scene, not the robot) | "recorded", and report lines.
    `record`: save this run as the reference in `ref` (healthy robot; low part at every free spot). Otherwise compare
    with `ref`, and write one picture per pose to `out` (top and base, masks against the reference's)."""
    ref, u = Path(ref), prim.u
    if record and (ref / "poses.json").exists():
        raise FileExistsError(f"{ref} already holds a reference: delete it first, on purpose")
    spots = [i for i in range(len(LOW_SPOTS)) if (record or (ref / f"s{i}_low_open_top.jpg").exists()) and _free(prim, i)]
    spots = spots if record else spots[:1]
    lines = [] if spots else ["height not checked: no free spot to lower at"]
    recs = []
    for label, *cmd in _plan(prim, spots):
        if not prim.goto(f"self-check: {label}", *cmd):
            prim._give_up("self-check")
            return "blocked", lines + [f"the filter blocked '{label}': the scene is in the way, not a robot fault"]
        top, base, _ = u._observe()
        state, _ = (u.robot.get_state_ws if u.use_ws else u.robot.get_state)()
        recs.append((label, cmd, top, cv2.cvtColor(base, cv2.COLOR_RGB2BGR), state))
    if record:
        ref.mkdir(parents=True, exist_ok=True)
        for label, _, top, base, _ in recs:
            cv2.imwrite(str(ref / f"{label}_top.jpg"), top)
            cv2.imwrite(str(ref / f"{label}_base.jpg"), base)
        (ref / "poses.json").write_text(json.dumps([dict(label=l, cmd=c, state=s) for l, c, _, _, s in recs], indent=1, default=float))
        return "recorded", lines + [f"{len(recs)} poses saved to {ref} (low spots {[LOW_SPOTS[i] for i in spots]})"]

    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
    fails, prev = [], None
    for k, (label, cmd, top, base, state) in enumerate(recs):
        rtop, rbase = cv2.imread(str(ref / f"{label}_top.jpg")), cv2.imread(str(ref / f"{label}_base.jpg"))
        m, r = _measure(top, base), _measure(rtop, rbase)
        near = {key: _near(m[key], r[key], NEAR_PX) for key in ("block", "claw", "base")} | {"frame": _near(m["frame"], r["frame"], FRAME_PX)}
        bad = [f"{key} {v:.3f} < {FRAME_MIN if key == 'frame' else NEAR_MIN}" for key, v in near.items() if v < (FRAME_MIN if key == "frame" else NEAR_MIN)]
        if not abs(m["tilt"] - r["tilt"]) <= TILT_DEG:  # nan (no block seen) fails too
            bad.append(f"block edge at {m['tilt']:+.1f} deg, reference {r['tilt']:+.1f}")
        if prev is not None and (np.array_equal(top, prev[0]) or np.array_equal(base, prev[1])):
            bad.append("same picture as the pose before (camera frozen or fetch failed)")
        if state is None:
            bad.append("no state from the robot")
        else:
            got = np.array([float(state[k]) for k in ("x_norm", "y_norm", "z_norm", "rotation", "claw_norm")]) / [1, 1, 1, 180, 1]
            if np.abs(got - cmd).max() > POSE_TOL:
                bad.append(f"robot reports {np.round(got, 3).tolist()} for {np.round(cmd, 3).tolist()}")
        lines.append(f"{label:14s} " + "  ".join(f"{key} {v:.3f}" for key, v in near.items()) + f"  tilt {m['tilt']:+.1f} (ref {r['tilt']:+.1f})")
        fails += [f"{label}: {b}" for b in bad]
        prev = top, base
        if out:
            t, b = _overlay(top, m["block"] | m["claw"], r["block"] | r["claw"]), _overlay(base, m["base"] | m["frame"], r["base"] | r["frame"])
            b = cv2.resize(b, (b.shape[1] * t.shape[0] // b.shape[0], t.shape[0]))
            pic = cv2.resize(np.hstack([t, b]), None, fx=0.5, fy=0.5)
            cv2.imwrite(str(Path(out) / f"{k:02d}_{label}{'_FAIL' if bad else ''}.jpg"), pic)
    return ("fail" if fails else "pass"), lines + (["FAILED:", *fails] if fails else [])
