"""Play primitives on the mock robot with a made-up scene (cell 20): the sequences run, and a blocked move gives up."""
import collections
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cloudgripper_wm import collision_filter as cf
from cloudgripper_wm.envs.cloudgripper_env import CloudGripperEnv
from cloudgripper_wm.envs.constants import PLAY_MAX_DELTA
from cloudgripper_wm.envs.filtered_wrapper import FilteredCloudGripperWrapper
from cloudgripper_wm.envs.robot_pool import RobotPool
from cloudgripper_wm.primitives import (NOTCH_PUSH_TRIES, OVER_ROPE_TIP_MM, PARK, PLAY_CARRY, PLAY_WEIGHTS, PUSH_CLEAR_MM, PUSH_TIP_MM,
                                       WIDE_OPEN, Primitives, _gap_mm, _pan_handle, z_for_tip)


def _obj(x, y, w_mm, h_mm, l_mm=None):
    a, b = w_mm / 2 * cf.MM, (l_mm or w_mm) / 2 * cf.MM
    return SimpleNamespace(status="on_glass", on=None, xy=(x, y), height_mm=h_mm, inferred_xy=None,
                           footprint_xy=np.array([[x - a, y - b], [x + a, y - b], [x + a, y + b], [x - a, y + b]]))


@pytest.fixture
def prim():
    keep = {k: getattr(cf, k) for k in list(cf.CELL20) + ["_FRAME"]}
    scene = {}
    tracker = SimpleNamespace(start=lambda b, t: scene, step=lambda b, t: scene, refresh=lambda b, t: scene)
    RobotPool.configure(["robot20"])
    env = FilteredCloudGripperWrapper(CloudGripperEnv(use_mock=True, dwell_time=0, reset_dwell_time=0, max_delta=PLAY_MAX_DELTA), tracker=tracker)
    env.reset()
    log, sent, step = [], [], env.step
    env.step = lambda a: (sent.append(np.array(a)), step(a))[1]  # every action that reaches the env
    p = Primitives(env, on_step=lambda label, blocked: log.append((label, blocked)))
    p.scene, p.log, p.sent = scene, log, sent
    yield p
    assert sent and (np.abs(sent) <= np.array(PLAY_MAX_DELTA) + 1e-6).all()  # only delta actions within the step limits
    env.close()
    for k, val in keep.items():
        setattr(cf, k, val)


def test_heights():
    assert abs(cf.tip_mm(z_for_tip(8.6)) - 8.6) < 1e-6 and z_for_tip(0.6) == 0.0


def test_push_and_give_up(prim):
    prim.scene.update(pan=_obj(0.3, 0.5, 20, 7))
    assert prim.push("pan", (1, 0), 30)
    u = prim.env.unwrapped
    assert u._target_pos[2] == 1.0 and any("push pan: push" in l and not b for l, b in prim.log)
    # the tips ended 30 mm past where they met the pan's back edge
    tips_x = u._target_pos[0] + cf.TOOL_OFFSET[0] + (np.array([[0, -1], [1, 0]]) @ cf.TOOL_OFFSET_TURNING)[0]
    assert abs(tips_x - (0.3 - 10 * cf.MM - cf.FINGER_MM[1] / 2 * cf.MM + 30 * cf.MM)) < 1e-3
    # something where the gripper has to come down behind the pan: not tried, nothing moves (play_1)
    prim.scene.update(slot=_obj(0.3 - 25 * cf.MM, 0.5, 20, 16))
    n = len(prim.log)
    assert not prim.push("pan", (1, 0), 30) and len(prim.log) == n
    # a cube beside where the tips come down: allowed by the filter (more than LOWER_MM away), but within PUSH_CLEAR_MM
    # (play_1: the tips came down on the white cube's corner); a little farther it fits
    plan = prim._push_plan("pan", (1, 0), 30)
    down = [*prim._cmd(plan[1], plan[0]).values(), z_for_tip(plan[3]), plan[0], prim.closed]
    for side_mm, fits in ((15.0, False), (19.0, True)):
        prim.scene.clear(); prim.scene.update(pan=_obj(0.3, 0.5, 20, 7), cube_blue=_obj(plan[1][0], 0.5 + side_mm * cf.MM, 10, 10))
        gap = min(_gap_mm(f, prim.scene["cube_blue"].footprint_xy) for f in cf.finger_polys(down))
        assert (cf.LOWER_MM < gap < PUSH_CLEAR_MM) != fits and prim._push_fits("pan", plan) == fits, gap
    # the slot shows up once the push is under way: the filter stops the lowering, it is given up and the arm raised
    del prim.scene["cube_blue"]
    prim.env._tracker.step = lambda b, t: (prim.scene.update(slot=_obj(0.3 - 25 * cf.MM, 0.5, 20, 16)), prim.scene)[1]
    prim.log.clear()
    assert not prim.push("pan", (1, 0), 30)
    assert prim.log[-2] == ("push pan: down", True) and "given up" in prim.log[-1][0] and u._target_pos[2] == 1.0


def test_push_over_the_rope(prim):
    """The rope right behind the slot (play_1; Kei: pushing the slot should be possible): the tips come down over the
    rope and meet the slot's solid end. Across the slot, where its notch is open at that height (a cube in it), they
    come down low behind the rope and push it into the slot."""
    prim.scene.update(slot=_obj(0.45, 0.45, 25.5, 16, 15), rope=_rope((0.45 - 25 * cf.MM, 0.3), np.pi / 2, length_mm=60))
    plan = prim._push_plan("slot", (1, 0), 15)
    assert plan[3] == OVER_ROPE_TIP_MM and prim._push_fits("slot", plan)
    assert prim.push("slot", (1, 0), 15) and ("push slot: push", False) in prim.log
    # behind a long side: through the rope (Kei, 2026-11-27): the tips come down low behind its far edge and push it
    # into the slot
    prim.scene["rope"] = _rope((0.3, 0.45 - 21 * cf.MM), 0.0, length_mm=60)
    plan = prim._push_plan("slot", (0, 1), 15)
    assert plan[3] == PUSH_TIP_MM and prim._push_fits("slot", plan)
    down = [*prim._cmd(plan[1], plan[0]).values(), z_for_tip(plan[3]), plan[0], prim.closed]
    assert max(f[:, 1].max() for f in cf.finger_polys(down)) < 0.45 - (25 + PUSH_CLEAR_MM) * cf.MM  # behind the rope


def test_play_varies_steps_and_merges_top_moves(prim):
    """Play (rng set; Kei, 2026-11-27): step sizes vary from move to move and step to step, within the env's limits
    (the fixture checks every action sent), each move still ends on its target; at the top height moves on different
    axes are sometimes one, never one that changes the height."""
    prim.rng, u = np.random.default_rng(0), prim.env.unwrapped
    counts = []
    for _ in range(6):
        n = len(prim.log)
        assert prim.goto("test: across", x=0.3, rot=0.2) and prim.goto("test: back", x=0.7, rot=0.6)
        counts.append(len(prim.log) - n)
        assert np.allclose(u._target_pos[[0, 3]], (0.7, 0.6), atol=1e-6)
    assert len(set(counts)) > 2, counts
    prim.scene.update(pan=_obj(0.3, 0.5, 20, 7))
    labels = []
    for _ in range(6):
        prim.log.clear()
        assert prim.push("pan", (1, 0), 30) and abs(u._target_pos[2] - 1.0) < 1e-6
        labels += [l for l, _ in prim.log]
    assert any(" + " in l for l in labels) and any(l == "push pan: close" for l in labels)
    # the gripper is binary (Kei): an open or close is one step, also inside a merged move
    prim.sent.clear()
    assert prim.goto("test: open", grip=prim.open) and prim.goto("test: close while moving", x=0.3, grip=prim.closed)
    grips = [a[4] for a in prim.sent if abs(a[4]) > 1e-6]
    assert len(grips) == 2 and all(abs(abs(g) - (prim.open - prim.closed)) < 1e-6 for g in grips), grips
    assert not any(" + " in l and ("up" in l or "down" in l) for l in labels), [l for l in labels if " + " in l]


def test_objects_against_the_frame(prim):
    x0, _, x1, _ = cf.GLASS
    # high-x frame: the slot is pushed back from over the frame; the pan is too low to be caught from there
    prim.scene.update(slot=_obj(x1 - 8 * cf.MM, 0.5, 15, 16, 25.5))
    assert prim.push("slot", (-1, 0), 30)
    prim.scene.clear(); prim.scene.update(pan=_obj(x1 - 11 * cf.MM, 0.5, 20, 7))
    assert not prim.push("pan", (-1, 0), 30)
    # low-x frame: the arm cannot get behind (its range ends at the glass edge), the push is given up
    prim.scene.clear(); prim.scene.update(slot=_obj(x0 + 8 * cf.MM, 0.5, 15, 16, 25.5))
    assert not prim.push("slot", (1, 0), 30)
    # 17 mm of glass behind the pan (robot20, 2026-11-26): a shorter run-up fits, the push is done
    prim.scene.clear(); prim.scene.update(pan=_obj(x0 + 27 * cf.MM, 0.5, 20, 7))
    assert prim.push("pan", (1, 0), 30)
    # a cube against either frame is clamped (tips along the frame, a little towards the glass) and slid in; the mock's
    # scene does not move, so the reset reports it still there
    for x in (x0 + 5 * cf.MM, x1 - 5 * cf.MM):
        prim.scene.clear(); prim.scene.update(cube_blue=_obj(x, 0.5, 10, 10))
        prim.log.clear()
        assert "cube_blue" in [p.split(":")[0] for p in prim.reset_objects()], prim.log[-3:]  # with the absent ones
        assert ("slide cube_blue: slide", False) in prim.log
        assert abs(prim.env.unwrapped._target_pos[3] - 0.5) < 1e-6  # tips along y, the frame's direction there


def test_grasp_place_slot_reset_rope(prim):
    prim.scene.update(cube_blue=_obj(0.3, 0.3, 10, 10), slot=_obj(0.6, 0.6, 15, 16, 25.5))
    assert prim.pick_place("cube_blue", (0.5, 0.3)) and prim.env.held == []
    # no room to open at the glass (the rope lies against the frame where a tip would open): the cube is let go
    # from the first height at which opening is allowed
    y0 = cf.GLASS[1]
    rope = _obj(0.3, y0 + 6 * cf.MM, 60, 8, 8)
    prim.scene.update(rope=rope)
    assert prim.grasp("cube_blue") and prim.goto("turn", rot=0.5) and prim.place((0.3, y0 + 24 * cf.MM)) and prim.env.held == []
    labels = [l for l, _ in prim.log]
    assert ("place: let go", True) in prim.log and "place: no room to open, up a step" in labels
    del prim.scene["rope"]
    # something beside the cube where one tip would come down: the tips take the cube's other side direction ...
    prim.scene.update(pan=_obj(0.3 + 14 * cf.MM, 0.3, 8, 7))
    assert prim.grasp("cube_blue") and abs(prim.env.unwrapped._target_pos[3] - 0.5) < 1e-6
    prim.place((0.3, 0.3))
    # ... and with no room either way the grasp is not tried at all
    prim.scene.update(cube_white=_obj(0.3, 0.3 + 14 * cf.MM, 8, 10))
    n = len(prim.log)
    assert not prim.grasp("cube_blue", make_room=False) and len(prim.log) == n
    # by default room is made first: one push the filter allows, then the grasp is tried once more (the mock
    # scene does not move, so that second try finds no room either)
    assert not prim.grasp("cube_blue")
    pushes = [l for l, b in prim.log[n:] if l.endswith(": push") and not b]
    assert pushes and len(set(pushes)) == 1 and not any("grasp" in l for l, _ in prim.log[n:])
    del prim.scene["pan"], prim.scene["cube_white"]
    assert prim.cube_into_slot("cube_blue") and prim.env.held == []
    let_go = [l for l, _ in prim.log].index("cube_blue into slot: let go")
    assert "down" in prim.log[let_go - 1][0] and not prim.log[let_go - 1][1]
    prim.scene.clear()
    prim.scene.update(cube_white=_obj(0.75, 0.75, 10, 10), pan=_obj(0.45, 0.5, 20, 7))  # only the cube is far from the centre
    prim.log.clear()
    prim.reset_objects()  # in a corner: no clamp fits, it is pushed out diagonally; the pan in the middle stays
    assert not any("grasp cube_white" in l for l, _ in prim.log) and ("push cube_white: up", False) in prim.log
    assert not any(l.startswith(("push pan", "slide pan", "grasp pan")) for l, _ in prim.log)
    rope = _obj(0.5, 0.2, 60, 8, 8); rope.footprint_xy = np.array([[x, 0.2 + s * 4 * cf.MM] for s in (-1, 1) for x in np.linspace(0.3, 0.7, 30)[::s]])
    prim.scene.clear(); prim.scene.update(rope=rope)
    assert prim.rope_drag(0, (0.0, 0.1)) and prim.env.held == []


def _pan(x, y, ang=0.0):
    """Pan.stl's outline: a 20 mm disc with an 8 mm wide handle reaching 21 mm out along `ang`; xy = centre of area."""
    t = np.linspace(0.45, 2 * np.pi - 0.45, 30)
    p = np.vstack([10 * np.c_[np.cos(t), np.sin(t)], [[21, -4], [21, 4]]])
    area = SimpleNamespace(c=np.array([3.4, 0.0]))  # centre of area, 3.4 mm towards the handle
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return SimpleNamespace(status="on_glass", on=None, xy=(x, y), height_mm=7, inferred_xy=None,
                           footprint_xy=np.array([x, y]) + (p - area.c) @ R.T * cf.MM)


def test_pan_is_carried_by_its_handle(prim):
    x0 = cf.GLASS[0]
    # round part against the low-x frame, handle towards the glass: no push can get behind it, the handle is grasped
    prim.scene.update(pan=_pan(x0 + 14 * cf.MM, 0.5))
    assert not prim.push("pan", (1, 0), 30)
    assert prim.grasp("pan") and prim.env.held == ["pan"]
    u = prim.env.unwrapped
    assert abs(u._target_pos[3] - 0.5) < 1e-6  # tips either side of a handle that points along x
    tips_x = u._target_pos[0] + cf.TOOL_OFFSET[0] + (np.array([[0, -1], [1, 0]]) @ cf.TOOL_OFFSET_TURNING)[0]
    assert abs(tips_x - (x0 + (14 - 3.4 + 15.5) * cf.MM)) < 1.5e-3  # 15.5 mm out from the round part's centre
    assert prim.place((0.45, 0.5)) and prim.env.held == []
    # handle towards the high-x frame and ending at it: held nearer the round part, where the tips clear the frame
    x1 = cf.GLASS[2]
    prim.scene.update(pan=_pan(x1 - (21 - 3.4) * cf.MM, 0.5))
    assert prim.grasp("pan") and prim.env.held == ["pan"]
    tips_x = u._target_pos[0] + cf.TOOL_OFFSET[0] + (np.array([[0, -1], [1, 0]]) @ cf.TOOL_OFFSET_TURNING)[0]
    assert tips_x < x1 - 7 * cf.MM and prim.place((0.45, 0.5))
    # the reset slides a pan at the frame in, clamped by its handle (never carried: it tilts)
    prim.scene.clear(); prim.scene.update(pan=_pan(x0 + 14 * cf.MM, 0.5))
    prim.log.clear()
    prim.reset_objects()
    assert ("slide pan: slide", False) in prim.log and not any(l.startswith("place") for l, _ in prim.log)


def test_pan_by_its_round_part(prim):
    """Handle over the high-x frame (prim_26): no handle spot fits. A lifting grasp gives up without moving (the round
    part does not lift, prim_27); a clamp takes the round part with the gripper opened to WIDE_OPEN, and the cell's
    range is back to the usual open afterwards."""
    x1, u = cf.GLASS[2], prim.env.unwrapped
    prim.scene.update(pan=_pan(x1 - (15 - 3.4) * cf.MM, 0.5))  # round part's centre 15 mm from the frame
    prim.goto("test: start", x=0.4)
    before = u._target_pos.copy()
    assert not prim.grasp("pan", make_room=False) and np.allclose(u._target_pos, before)
    grips, step = [], prim.env.step
    prim.env.step = lambda a: (grips.append(float(u._target_pos[4] + a[4])), step(a))[1]
    assert prim.grasp("pan", lift=False) and prim.env.held == ["pan"] and u._target_pos[2] == 0.0
    tips = np.concatenate(cf.finger_polys(u._target_pos)).mean(0)
    assert abs(max(grips) - WIDE_OPEN) < 1e-6 and np.linalg.norm(tips - (x1 - 15 * cf.MM, 0.5)) < 1.5e-3
    assert tuple(u._gripper_range) == (prim.closed, prim.open) and prim.open < WIDE_OPEN


def _rope(p0, ang, length_mm=100, w_mm=8):
    """A straight rope from p0 along ang: its outline as the tracker gives it (few points on the straight sides)."""
    d, n = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
    p0, h = np.asarray(p0, float), w_mm / 2 * cf.MM
    p1 = p0 + d * length_mm * cf.MM
    fp = np.array([p0 - n * h, p1 - n * h, p1 + d * h, p1 + n * h, p0 + n * h, p0 - d * h])
    return SimpleNamespace(status="on_glass", on=None, xy=tuple((p0 + p1) / 2), height_mm=8, inferred_xy=None, footprint_xy=fp)


def test_rope_drag_near_the_frame(prim):
    y0 = cf.GLASS[1]
    # the end 4 mm from the low-y frame, the rope going off at 45 deg: near the end a tip would come down on the
    # frame (robot20, 2026-11-26), farther in the tips fit
    prim.scene.update(rope=_rope((0.3, y0 + 8 * cf.MM), np.pi / 4))
    end = int(np.argmin([q[1] for q, _ in prim._rope_ends()]))
    assert prim.rope_drag(end, (0.0, 0.05)) and prim.env.held == []
    over = [l for l, _ in prim.log if "over the rope" in l][0]
    assert "12 mm" not in over, over
    # along the frame all the way: no spot fits, nothing moves
    prim.scene.update(rope=_rope((0.3, y0 + 9 * cf.MM), 0.0)); n = len(prim.log)
    assert not prim.rope_drag(0, (0.0, 0.05)) and len(prim.log) == n
    # the slot on the glass side of a short rope, the frame on the other (play_1): no spot fits; towards the centre
    # the push would come down on the rope, along the rope it fits and frees it (Kei, 2026-11-27)
    rope, slot = _rope((0.3, y0 + 22 * cf.MM), 0.0, length_mm=50), _obj(0.3 + 25 * cf.MM, y0 + 41.5 * cf.MM, 25.5, 16, 15)
    prim.scene.update(rope=rope, slot=slot)
    assert isinstance(prim._rope_spot(0), list)
    n, d = prim._rope_room(prim._rope_spot(0), 0)
    assert n == "slot" and abs(d[0]) > 0.5, d  # along the rope, not towards the centre (+y)
    pushed = lambda b, t: (any(l == "push slot: push" for l, _ in prim.log) and prim.scene.update(slot=_obj(0.6, 0.6, 25.5, 16, 15)), prim.scene)[1]
    prim.env._tracker.step = pushed
    prim.log.clear()
    assert prim.rope_drag(0, (0.0, 0.03))
    labels = [l for l, _ in prim.log]
    assert labels.index("push slot: push") < next(i for i, l in enumerate(labels) if "over the rope" in l)


def test_rope_held_after_closing_across_it(prim):
    """play_1 episode 7: the filter's clamp test missed the rope (its mask breaks up under the fingers); the drag said
    "nothing between the tips" and the closed gripper rose with the rope in it. Closed across the rope at the glass it
    is held: the drag runs and lets go."""
    prim.scene.update(rope=_rope((0.3, 0.45), 0.0, length_mm=60))
    import cloudgripper_wm.collision_filter as filt
    clamped = filt._clamped
    filt._clamped = lambda *a: []  # the clamp test misses
    try:
        assert prim.rope_drag(0, (0.0, 0.03)) and prim.env.held == []
    finally:
        filt._clamped = clamped
    labels = [l for l, _ in prim.log]
    assert "rope drag: drag" in labels and labels.index("rope drag: drag") < labels.index("rope drag: let go")


def test_rope_dragged_away_from_a_point(prim):
    """Clearing room (Kei, 2026-11-27: clear the space near the pan): the rope is taken at the spot nearest the point and
    dragged away from it."""
    prim.scene.update(rope=_rope((0.3, 0.45), 0.0, length_mm=80))
    point = (0.3 + 60 * cf.MM, 0.45 - 20 * cf.MM)  # below the rope, near its far end
    spot = prim._rope_spot(0, near=point)
    assert abs(spot[0][0] - point[0]) < 5 * cf.MM, spot  # the spot nearest the point (spots every 8 mm)
    assert prim.rope_drag(0, (20 * cf.MM, 0.0), away_from=point)
    i = max(i for i, (l, _) in enumerate(prim.log) if l == "rope drag: close")
    ys = [a[1] for a in prim.sent[-len(prim.log) + i + 1:]]
    assert sum(ys) > 0  # dragged up, away from the point below


def test_rope_twist(prim):
    """Kei, 2026-11-27: clamp the rope and turn the gripper (the other way when the turn would leave the range), with or
    without a drag; the first grasp spot is ROPE_END_MM in from the end (thinner at the tip)."""
    prim.scene.update(rope=_rope((0.3, 0.45), 0.0, length_mm=60))
    prim.log.clear()
    assert prim.rope_drag(0, (0.0, 0.0), twist=0.3) and prim.env.held == []
    labels = [l for l, _ in prim.log]
    assert "rope drag: twist" in labels and any("20 mm from the end" in l for l in labels), labels
    turned = [p for p in prim.sent if p[3] != 0]  # the twist's steps turn while the tips are down
    assert turned
    for together, want in ((True, ["rope drag: drag + twist"]), (False, ["rope drag: drag", "rope drag: twist"])):
        prim.log.clear()
        assert prim.rope_drag(1, (0.0, 0.03), twist=0.2, together=together)
        labels = [l for l, _ in prim.log]
        assert all(w in labels for w in want) and (together or "rope drag: drag + twist" not in labels), labels


def test_slot_is_carried_across_its_width(prim):
    x0 = cf.GLASS[0]
    # an end against the low-x frame: nothing reaches behind it, but the tips fit either side of its 15 mm width
    prim.scene.update(slot=_obj(x0 + 14 * cf.MM, 0.5, 25.5, 16, 15))
    assert not prim.push("slot", (1, 0), 30)
    prim.log.clear()
    prim.reset_objects()  # a push first (Kei), which cannot get behind it here; then clamped across its width and slid
    assert ("slide slot: slide", False) in prim.log
    u = prim.env.unwrapped
    assert abs(u._target_pos[3] - 0.5) < 1e-6 and prim.env.held == []  # tips were separated along y, its width


@pytest.mark.parametrize("way", ["push", "drop", "swing", "grasp"])
def test_cube_out_of_slot(prim, way):
    slot = _obj(0.4, 0.5, 15, 16, 25.5)  # long along y: the notch runs along x
    cube = SimpleNamespace(status="stacked", on="slot", height_mm=10, footprint_xy=None, inferred_xy=(0.4, 0.5), in_notch=True)
    prim.scene.update(slot=slot, cube_blue=cube)
    # the mock scene does not move: as long as the tracker still reports it in the slot, it is not done (a push is
    # tried NOTCH_PUSH_TRIES times)
    plans, plan = [], prim._notch_push
    prim._notch_push = lambda c: (plans.append(c), plan(c))[1]
    assert not prim.cube_out_of_slot("cube_blue", way) and prim.log[-1][0] == "cube_blue out of slot: back home to look"
    assert len(plans) == (NOTCH_PUSH_TRIES if way == "push" else 0)
    prim._notch_push = plan
    assert not any(b for _, b in prim.log), [l for l, b in prim.log if b]  # nothing blocked
    # looking again from home finds it on the glass: done
    prim.env._tracker.refresh = lambda b, t: (prim.scene.update(cube_blue=_obj(0.45, 0.5, 10, 10)), prim.scene)[1]
    prim.scene["cube_blue"] = cube
    assert prim.cube_out_of_slot("cube_blue", way) and prim.env.held == []
    prim.scene["cube_blue"] = cube
    # beside the slot rather than in its notch: the slot grows by a cube, nothing is tried
    cube.in_notch = False; n = len(prim.log)
    assert not prim.cube_out_of_slot("cube_blue", way) and len(prim.log) == n


def test_cube_pushed_out_after_moving_the_slot(prim):
    x0 = cf.GLASS[0]
    notch = lambda x: dict(slot=_obj(x, 0.5, 15, 16, 25.5),  # long along y: the notch runs along x, towards the frame
                           cube_blue=SimpleNamespace(status="stacked", on="slot", height_mm=10, footprint_xy=None, inferred_xy=(x, 0.5), in_notch=True))
    prim.scene.update(notch(x0 + 20 * cf.MM))
    assert isinstance(prim._notch_push("cube_blue"), str)  # the tip would come down over the frame to enter the notch
    looks = iter([notch(0.4), dict(cube_blue=_obj(0.45, 0.5, 10, 10))])  # after the carry; after the push
    prim.env._tracker.refresh = lambda b, t: (prim.scene.update(next(looks)), prim.scene)[1]
    n = len(prim.log)
    assert not prim.cube_out_of_slot("cube_blue", "push") and len(prim.log) == n  # by default the slot is not carried
    assert prim.cube_out_of_slot("cube_blue", "push", move_slot=True)
    labels = [l for l, _ in prim.log]
    assert labels.index("grasp slot: close") < labels.index("cube_blue out of slot (push): push")


def test_grasp_checks_the_lift_and_place_checks_for_a_drop(prim):
    """prim_21 (2026-11-26): the slot slipped out on the lift, "holding slot" stayed, and the arm was lowered onto it."""
    u = prim.env.unwrapped
    cube = _obj(0.5, 0.5, 10, 10)
    prim.scene.update(cube_blue=cube)

    def mask(y):
        a = np.zeros((720, 1280), bool)
        a[y:y + 40, 600:640] = True
        return a
    cube.mask_top = mask(400)
    # the top view never sees it rise: not lifted, the gripper opens and nothing is held
    assert not prim.grasp("cube_blue") and prim.env.held == [] and prim.log[-1][0] == "grasp cube_blue: not lifted, opening"
    # it rises with the lift: held
    prim.env._tracker.step = lambda b, t: (setattr(cube, "mask_top", mask(300 if prim.env.held and u._target_pos[2] > 0.9 else 400)), prim.scene)[1]
    assert prim.grasp("cube_blue") and prim.env.held == ["cube_blue"]
    labels = [l for l, _ in prim.log]
    assert labels.index("grasp cube_blue: ease") < labels.index("grasp cube_blue: close again")
    # carried 0.1 in x, which should move it 100 px in this made-up top view; it stays where it was: dropped
    prim.env._tracker.top_px = lambda xy: np.array([0.0, xy[0] * 1000])
    n = len(prim.log)
    assert not prim.place((0.6, 0.5)) and prim.env.held == []
    assert not any(l.startswith("place: down") for l, _ in prim.log[n:])



def test_scene_check(prim):
    """Fit: every object on the glass or a cube in the notch, none at the frame. The arm looks from PARK."""
    prim.scene.update(cube_blue=_obj(0.4, 0.4, 15, 15), cube_white=_obj(0.5, 0.6, 15, 15), slot=_obj(0.6, 0.4, 15, 16, 25.5),
                      pan=_obj(0.35, 0.6, 20, 7), rope=_obj(0.45, 0.3, 60, 8, 8))
    assert prim.scene_check() == [] and np.allclose(prim.u._target_pos[:3], (*PARK, 1.0))
    prim.scene["cube_blue"] = SimpleNamespace(status="stacked", on="slot", in_notch=True, height_mm=15, inferred_xy=None)
    assert prim.scene_check() == []
    prim.scene["cube_white"] = _obj(0.05, 0.5, 15, 15)  # against the low-x frame
    prim.scene["pan"] = SimpleNamespace(status="unaccounted", on=None, height_mm=7, inferred_xy=None)
    prim.scene["cube_blue"].in_notch = False  # on the slot's wall, not in the notch
    assert [p.split(":")[0] for p in prim.scene_check()] == ["cube_blue", "cube_white", "pan"]
    prim.scene.update(cube_blue=_obj(0.4, 0.4, 15, 15), cube_white=_obj(0.5, 0.6, 15, 15), pan=_obj(0.35, 0.6, 20, 7))
    prim.scene["rope"] = _obj(0.4, 0.06, 60, 8, 8)  # along the low-y frame, one end well inside
    prim.scene["rope"].footprint_xy[1] = (0.4, 0.3)
    assert prim.scene_check() == []
    prim.scene["rope"] = _obj(0.4, 0.06, 60, 8, 8)  # all of it along the frame
    assert [p.split(":")[0] for p in prim.scene_check()] == ["rope"]


def test_random_primitive(prim):
    """The object first, evenly among those with a candidate, then its kind by PLAY_WEIGHTS; targets keep clear of the
    frame."""
    rng = np.random.default_rng(0)
    prim.scene.update(cube_blue=_obj(0.4, 0.4, 15, 15), cube_white=_obj(0.5, 0.6, 15, 15), slot=_obj(0.6, 0.4, 15, 16, 25.5),
                      pan=_obj(0.3, 0.6, 20, 7), rope=_obj(0.45, 0.25, 60, 8, 8))
    assert prim.goto("test: a step", x=0.4)  # the fixture wants at least one step sent
    picks = [prim.random_primitive(rng) for _ in range(150)]
    kinds = {k: sum(p[0] == k for p in picks) for k in PLAY_WEIGHTS}
    # five objects, each about a fifth; for a cube, pick-and-place or into the slot (2) before a push (1)
    objs = collections.Counter("rope" if k == "rope_drag" else a[0] for k, a in picks)
    assert set(objs) == {"cube_blue", "cube_white", "slot", "pan", "rope"} and min(objs.values()) > 0.1 * len(picks), objs
    cube = collections.Counter(k for k, a in picks if a[0] in ("cube_blue", "cube_white"))
    assert kinds["cube_out_of_slot"] == 0 and cube["cube_into_slot"] + cube["pick_place"] > 2 * cube["push"] and kinds["pick_place"] > 0
    x0, y0, x1, y1 = cf.GLASS
    for k, a in picks:
        if k == "pick_place":
            assert a[0] in PLAY_CARRY and x0 + 0.1 < a[1][0] < x1 - 0.1 and y0 + 0.1 < a[1][1] < y1 - 0.1
        if k == "push":  # only pushes the filter allows, with room where the tips come down (play_1: two were stopped)
            assert prim._push_fits(a[0], prim._push_plan(a[0], a[1], a[2], say=False))
    assert kinds["push"] > 0
    # the rope along the frame where no spot fits the open tips and nothing can be pushed aside: never drawn (play_1)
    prim.scene["rope"] = _rope((0.3, cf.GLASS[1] + 9 * cf.MM), 0.0)
    assert isinstance(prim._rope_spot(0), list) and not any(k == "rope_drag" for k, _ in (prim.random_primitive(rng) for _ in range(20)))
    prim.scene["rope"] = _obj(0.45, 0.25, 60, 8, 8)
    prim.scene["cube_blue"] = SimpleNamespace(status="stacked", on="slot", in_notch=True, height_mm=15, inferred_xy=None)
    picks = [prim.random_primitive(rng) for _ in range(60)]
    assert ("cube_out_of_slot", ("cube_blue", "push")) in picks and not any(k == "cube_into_slot" for k, _ in picks)
    prim.scene["pan"] = _obj(0.08, 0.6, 20, 7)  # by the frame: never pushed further towards it
    assert not any(k == "push" and a[0] == "pan" and a[1][0] < 0 for k, a in (prim.random_primitive(rng) for _ in range(60)))
    # nor by a push of what touches it (play_1 episode 8: a slot push took the pan along to the frame)
    prim.scene["slot"] = _obj(0.08 + 19 * cf.MM, 0.6, 15, 16, 25.5)
    assert not any(k == "push" and a[0] == "slot" and a[1][0] < -0.3 for k, a in (prim.random_primitive(rng) for _ in range(60)))


def test_slide_and_the_pan_at_the_frame(prim):
    """slide: clamp at the bottom, move along the glass, let go, up. The reset slides a pan it can neither carry nor
    push (prim_27: the round part did not lift) out from the frame first."""
    x1, u = cf.GLASS[2], prim.env.unwrapped
    prim.scene.update(pan=_pan(x1 - (15 - 3.4) * cf.MM, 0.5))
    prim.goto("test: start", x=0.4)
    assert prim.slide("pan", (-1, 0), 20)
    labels = [l for l, _ in prim.log]
    i = labels.index("slide pan: slide")
    assert "grasp pan: close again" in labels[:i] and not any("grasp pan: lift" in l for l in labels)
    assert u._target_pos[2] == 1.0 and prim.env.held == [] and np.isclose(u._target_pos[4], prim.open)
    prim.log.clear()
    pick = prim.pick_place
    prim.pick_place = lambda n, xy: False if n == "pan" else pick(n, xy)  # as on the robot: the round part does not lift
    prim.reset_objects()
    assert any(l.startswith("slide pan") for l, _ in prim.log)
    t = np.linspace(0, 2 * np.pi, 40)
    assert _pan_handle(0.4 + 10 * cf.MM * np.c_[np.cos(t), np.sin(t)])[1] is None  # no handle seen: no direction, no NaN


def test_free_spot_keeps_clear_of_outlines(prim):
    """prim_28: a spot 22 mm from the pan's corners was next to its long edge, and the white cube went onto the pan."""
    prim.goto("test: a step", x=0.4)
    long = _obj(0.44, 0.45, 60, 7, 6)  # a long thin outline across the centre: corners far, edge close
    prim.scene.update(pan=long)
    p = prim._free_spot("cube_white")
    edge = -cv2.pointPolygonTest(np.float32(long.footprint_xy / cf.MM), tuple(map(float, p / cf.MM)), True)
    assert edge > 22
    prim.scene.update(pan=_obj(0.44, 0.45, 110, 7, 110))  # covers the glass: no free place, and nothing is carried
    assert prim._free_spot("cube_white") is None and not prim.pick_place("cube_blue", None)


def test_given_up_while_holding_puts_it_back(prim):
    """prim_28: lowering the white cube onto the pan was blocked, and the give-up dropped it from the top height onto
    the pan. Now it goes back where it was taken and is let go there."""
    prim.scene.update(cube_white=_obj(0.35, 0.4, 15, 15), pan=_obj(0.6, 0.6, 30, 7, 30))
    assert prim.grasp("cube_white") and prim.env.held == ["cube_white"]
    taken = prim.env.unwrapped._target_pos[:2].copy()
    prim.log.clear()
    assert not prim.place((0.6, 0.6))  # over the pan: carried there, the lowering is blocked
    labels = [l for l, _ in prim.log]
    assert "place: given up, back where it was taken" in labels and not any("dropping" in l for l in labels)
    i = labels.index("place: given up, back where it was taken")
    assert np.allclose(prim.env.unwrapped._target_pos[:2], taken) and prim.env.held == [] and "let go" in " ".join(labels[i:])
