"""Self-check of a play cell before and during data collection (Kei, 2026-11-27). MOVES THE ROBOT.

The arm visits a fixed list of poses (`_plan`) through the filtered env like every other move, never below height
0.35 (Kei). At each pose it takes both pictures and compares them with a reference recorded on the healthy robot
at the same pose:
- every axis got there and the gripper looks as it did: in the top view, the yellow block (the gripper's fixed
  anchor; it moves with x, y and height, not with the rotation) and the claw under it (the finger linkages; the
  cable's thin yellow stripe is opened away); in the base view, all the yellow. Each mask must lie within NEAR_PX
  of the reference's (`_near`: closing twice leaves the fingers a few px apart);
- the gripper hangs straight: the angle of the block's bottom edge matches the reference within TILT_DEG (+2.3 deg
  at x 0.5, y 0.5, top height on 2026-11-26, open, half and closed alike; Kei: a broken one tilts about 30 deg);
- the base camera was not moved: the navy frame's shift against the reference's, by phase correlation, is at most
  FRAME_SHIFT_PX (a moved top camera shows as the block off at every pose);
- the cameras deliver: no picture is the same array as the previous pose's (the env reuses the last frame when a
  fetch fails, which looks like an arm that did not move).
The robot's reported pose is printed beside each pose, not judged: on 2026-11-23 its x, y lagged the arm, cause not
established (Kei: network lag at that moment, open-loop error building up, or the stream of commands); these logs
will tell whether it can be trusted. Motor currents are not judged either (Kei: never used).

The height is checked at the reference x, y, down to LOW_Z: there the tips (about 19 mm) clear a cube and the slot,
so only a stack or a cube in the slot under the gripper keeps it from running. At 0.35 the tips are below a cube's
top and in the first recording (2026-11-27) none of three spots was free.

The reference lives in REF_DIR/<robot>, committed to git (Kei: safe from accidental deletion). Recording adds the
poses the reference lacks and never replaces one.

Healthy numbers, 2026-11-27 (first check against the reference): block 1.0, claw 0.987 and up, base 0.945 and up,
tilt within 0.1 deg of the reference, reported x, y within 0.001; frame shift at most 0.11 px over the ten
reference pictures, and a 1 px shift reads 1 px. Distinguishing: claw half-open against closed 0.81, base yellow
open against closed 0.59 (2026-11-26 pictures; the base view sees only ~270 yellow px at the top height).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from cloudgripper_wm import collision_filter as cf
from cloudgripper_wm.primitives import wait_scale

C, R = (0.5, 0.5), 1.0  # reference pose: middle of the arm's range, rotation 180 deg (fingers side-on to the top camera)
HALF = 0.45            # half-open gripper command
LOW_Z = 0.7            # lowest height in the check: tips ~19 mm, clear of a cube (15) and the slot (16) with CLEAR_MM
YELLOW = ((18, 120, 120), (35, 255, 255))  # HSV ranges, cell 20
NAVY = ((100, 70, 30), (125, 255, 160))    # the frame from below (S ~110, V ~70: the bright blue edge alone was too thin)
FRAME_PIECE_PX = 5000  # frame pieces at least this large: it is two parts with a small gap (Kei); the blue cube is smaller
NEAR_PX, NEAR_MIN, BASE_MIN = 6, 0.92, 0.85
FRAME_SHIFT_PX = 5.0   # Kei, 2026-11-27: no tight margin; healthy 0.1 to 0.3 px
TILT_DEG = 10.0        # Kei, 2026-11-27
RETAKES, RETAKE_S = 2, 1.0  # a failing pose is pictured again this often, this far apart, before it counts: a false
                            # alarm stops the collection, worse than a missed fault (Kei, 2026-11-27)
REF_DIR = Path(__file__).parent / "self_check_ref"  # one reference per robot, in git


def _mask(img, hsv_range) -> np.ndarray:
    return cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), *hsv_range) > 0


def _split(top) -> tuple[np.ndarray, np.ndarray]:
    """Top view: the gripper's yellow block (the largest yellow piece) and the claw under it."""
    y = _mask(top, YELLOW)
    n, lab, st, _ = cv2.connectedComponentsWithStats(y.astype(np.uint8))
    if n < 2:
        return y, y
    i = 1 + st[1:, 4].argmax()
    blk, x, w = lab == i, st[i, 0], st[i, 2]
    claw = cv2.morphologyEx((y & ~blk).astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)) > 0
    claw[:np.where(blk.any(1))[0].max()] = False
    claw[:, :max(0, x - w // 3)] = claw[:, x + w + w // 3:] = False
    return blk, claw


def _frame(base) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(_mask(base, NAVY).astype(np.uint8))
    return np.isin(lab, 1 + np.where(st[1:, 4] > FRAME_PIECE_PX)[0])


def _shift(a, b) -> float:
    """How far (px) mask b lies from mask a, by phase correlation; inf when either is empty."""
    if not a.any() or not b.any():
        return float("inf")
    return float(np.hypot(*cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))[0]))


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
    return dict(block=blk, claw=claw, base=_mask(base, YELLOW), frame=_frame(base), tilt=_tilt(blk))


def _plan(prim, low: bool) -> list[tuple]:
    """(label, x, y, z, rot, grip) for each pose: every axis both ways at the top height, then down and up."""
    o, c, (x, y), up = prim.open, prim.closed, C, 1.0
    plan = [("open", x, y, up, R, o), ("closed", x, y, up, R, c), ("half", x, y, up, R, HALF),
            ("x+", x + .1, y, up, R, o), ("x-", x - .1, y, up, R, o), ("y+", x, y + .1, up, R, o), ("y-", x, y - .1, up, R, o),
            ("rot135", x, y, up, R - .25, o), ("rot90", x, y, up, R - .5, o), ("open_again", x, y, up, R, o)]
    return plan + ([("down_open", x, y, LOW_Z, R, o), ("down_closed", x, y, LOW_Z, R, c), ("up", x, y, up, R, c)] if low else [])


def _free_below(prim) -> bool:
    """Does the filter let the gripper down to LOW_Z at the reference x, y, and close there, as the scene is now?"""
    over, low = [*C, 1.0, R, prim.open], [*C, LOW_Z, R, prim.open]
    return cf.check(over, low, prim.env.states).allowed and cf.check(low, [*low[:4], prim.closed], prim.env.states).allowed


def _reported(state, cmd) -> str:
    """The robot's reported pose minus the commanded one (x, y, z, rotation, gripper): printed, not judged."""
    if not state:
        return "  reported: none"
    got = [float(state[k]) for k in ("x_norm", "y_norm", "z_norm")] + [float(state["rotation"]) / 180, float(state["claw_norm"])]
    return "  reported off " + " ".join(f"{a}{g - c:+.3f}" for a, g, c in zip("xyzrg", got, cmd))


def _overlay(img, cur, ref) -> np.ndarray:
    """Reference only magenta, this check only cyan, both white."""
    o = img // 2
    o[ref & ~cur], o[cur & ~ref], o[cur & ref] = (255, 0, 255), (255, 255, 0), (255, 255, 255)
    return o


def _judge(label, top, base, prev, ref: Path) -> tuple[list[str], dict, dict, dict]:
    """(what is wrong, the numbers, this pose's masks, the reference's) for one picture pair."""
    m, r = _measure(top, base), _measure(cv2.imread(str(ref / f"{label}_top.jpg")), cv2.imread(str(ref / f"{label}_base.jpg")))
    nums = {key: _near(m[key], r[key], NEAR_PX) for key in ("block", "claw", "base")}
    nums |= dict(frame_px=_shift(r["frame"], m["frame"]), tilt=m["tilt"], ref_tilt=r["tilt"])
    bad = [f"{key} {nums[key]:.3f} < {BASE_MIN if key == 'base' else NEAR_MIN}" for key in ("block", "claw", "base")
           if nums[key] < (BASE_MIN if key == "base" else NEAR_MIN)]
    if not nums["frame_px"] <= FRAME_SHIFT_PX:
        bad.append(f"base camera moved: frame {nums['frame_px']:.1f} px off")
    if not abs(m["tilt"] - r["tilt"]) <= TILT_DEG:  # nan (no block seen) fails too
        bad.append(f"block edge at {m['tilt']:+.1f} deg, reference {r['tilt']:+.1f}")
    if prev is not None and (np.array_equal(top, prev[0]) or np.array_equal(base, prev[1])):
        bad.append("same picture as the pose before (camera frozen or fetch failed)")
    return bad, nums, m, r


def self_check(prim, ref=None, out=None, record: bool = False) -> tuple[str, list[str]]:
    """Run the check with `prim` (Primitives on a FilteredCloudGripperWrapper after reset). Returns the status,
    "pass" | "fail" | "blocked" (the filter stopped a move: the scene, not the robot) | "recorded", and report lines.
    `record`: add this run's poses that `ref` (default REF_DIR/<robot>) lacks; healthy robot only. Otherwise compare
    with `ref`.

    `out` keeps everything for diagnosis (Kei, 2026-11-27): scene_start.npz (the tracker's states with their masks,
    `save_states`), raw/ (every picture taken, retakes included, full size), log.jsonl (one line per picture: pose,
    take, command, reported state, the numbers, what was wrong), one overlay per pose (top and base, reference-only
    masks magenta, this check's cyan, both white; "_FAIL" where the pose failed) and report.txt."""
    ref = Path(ref) if ref else REF_DIR / prim.env._name
    if out:
        out = Path(out)
        (out / "raw").mkdir(parents=True, exist_ok=True)
        from cloudgripper_wm.perception.scene_tracker import save_states  # here: the module loads torch
        save_states(out / "scene_start.npz", prim.env.states)
    status, lines = _run(prim, ref, out, record)
    if out:
        (out / "report.txt").write_text("\n".join(lines + [f"== self-check: {status.upper()}"]) + "\n")
    return status, lines


def _run(prim, ref: Path, out, record: bool) -> tuple[str, list[str]]:
    u = prim.u
    low = _free_below(prim)
    if record and not low:
        return "blocked", [f"something tall under the gripper at {C}: clear it, then record"]
    lines = [] if low else [f"height not checked: something tall under the gripper at {C}"]
    log = open(out / "log.jsonl", "a") if out else None
    recs, fails, prev = [], [], None
    try:
        for k, (label, *cmd) in enumerate(_plan(prim, low)):
            if not prim.goto(f"self-check: {label}", *cmd):
                prim._give_up("self-check")
                return "blocked", lines + [f"the filter blocked '{label}': the scene is in the way, not a robot fault"]
            m = None
            for take in range(RETAKES + 1):
                if take:
                    time.sleep(RETAKE_S * wait_scale(u))
                top, base, _ = u._observe()
                base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
                state, _ = (u.robot.get_state_ws if u.use_ws else u.robot.get_state)()
                bad, nums = [], {}
                if not record and not (ref / f"{label}_top.jpg").exists():
                    bad = [f"not in the reference {ref} (record it)"]
                elif not record:
                    bad, nums, m, r = _judge(label, top, base, prev, ref)
                if out:
                    cv2.imwrite(str(out / "raw" / f"{k:02d}_{label}_{take}_top.jpg"), top)
                    cv2.imwrite(str(out / "raw" / f"{k:02d}_{label}_{take}_base.jpg"), base)
                    log.write(json.dumps(dict(pose=k, label=label, take=take, t=time.time(), cmd=cmd, state=state, **nums, bad=bad),
                                         default=float) + "\n")
                if record or not bad or m is None:
                    break
            prev = top, base
            if record:
                recs.append((label, cmd, top, base, state))
                continue
            if m is None:
                lines.append(f"{label:12s} not in the reference")
            else:
                lines.append(f"{label:12s} " + "  ".join(f"{key} {nums[key]:.3f}" for key in ("block", "claw", "base"))
                             + f"  frame {nums['frame_px']:.1f} px  tilt {nums['tilt']:+.1f} (ref {nums['ref_tilt']:+.1f})"
                             + _reported(state, cmd) + (f"  (pictured {take + 1} times)" if take else ""))
            fails += [f"{label}: {b}" for b in bad]
            if out and m is not None:
                t, b = _overlay(top, m["block"] | m["claw"], r["block"] | r["claw"]), _overlay(base, m["base"] | m["frame"], r["base"] | r["frame"])
                b = cv2.resize(b, (b.shape[1] * t.shape[0] // b.shape[0], t.shape[0]))
                cv2.imwrite(str(out / f"{k:02d}_{label}{'_FAIL' if bad else ''}.jpg"), cv2.resize(np.hstack([t, b]), None, fx=0.5, fy=0.5))
    finally:
        if log:
            log.close()
    if not record:
        return ("fail" if fails else "pass"), lines + (["FAILED:", *fails] if fails else [])
    ref.mkdir(parents=True, exist_ok=True)
    poses = json.loads((ref / "poses.json").read_text()) if (ref / "poses.json").exists() else []
    new = [r for r in recs if not (ref / f"{r[0]}_top.jpg").exists()]
    for label, cmd, top, base, state in new:
        cv2.imwrite(str(ref / f"{label}_top.jpg"), top)
        cv2.imwrite(str(ref / f"{label}_base.jpg"), base)
        poses.append(dict(label=label, cmd=cmd, state=state))
    (ref / "poses.json").write_text(json.dumps(poses, indent=1, default=float))
    return "recorded", [f"{len(new)} new poses saved to {ref} ({', '.join(r[0] for r in new) or 'none'}); existing ones kept"]
