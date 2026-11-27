"""Play primitives on the mock robot with a made-up scene (cell 20): the sequences run, and a blocked move gives up."""
from types import SimpleNamespace

import numpy as np
import pytest

from cloudgripper_wm import collision_filter as cf
from cloudgripper_wm.envs.cloudgripper_env import CloudGripperEnv
from cloudgripper_wm.envs.constants import PLAY_MAX_DELTA
from cloudgripper_wm.envs.filtered_wrapper import FilteredCloudGripperWrapper
from cloudgripper_wm.envs.robot_pool import RobotPool
from cloudgripper_wm.primitives import NOTCH_PUSH_TRIES, PARK, PLAY_CARRY, PLAY_WEIGHTS, WIDE_OPEN, Primitives, _pan_handle, z_for_tip


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
    p.scene, p.log = scene, log
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
    # something where the gripper has to come down behind the pan: the primitive is given up and the arm raised
    prim.scene.update(slot=_obj(0.3 - 25 * cf.MM, 0.5, 20, 16))
    prim.log.clear()
    assert not prim.push("pan", (1, 0), 30)
    assert prim.log[-2] == ("push pan: down", True) and "given up" in prim.log[-1][0] and u._target_pos[2] == 1.0


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
    # a cube against either frame is picked up (tips along the frame, a little towards the glass) and put down
    for x in (x0 + 5 * cf.MM, x1 - 5 * cf.MM):
        prim.scene.clear(); prim.scene.update(cube_blue=_obj(x, 0.5, 10, 10))
        assert prim.reset_objects() == {"cube_blue": True}, prim.log[-3:]
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
    assert prim.reset_objects() == {"cube_white": True}  # in a corner: no grasp fits, it is pushed out diagonally
    assert not any("grasp cube_white" in l for l, _ in prim.log) and prim.log[-1][0] == "push cube_white: up"
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
    # the reset takes that way for a pan far from the centre
    prim.scene.clear(); prim.scene.update(pan=_pan(x0 + 14 * cf.MM, 0.5))
    assert prim.reset_objects() == {"pan": True} and any("grasp pan: close" in l for l, _ in prim.log)


def test_pan_by_its_round_part(prim):
    """Handle over the high-x frame (prim_26): no handle spot fits, so the round part is taken with the gripper opened
    to WIDE_OPEN, and the cell's range is back to the usual open afterwards."""
    x1, u = cf.GLASS[2], prim.env.unwrapped
    prim.scene.update(pan=_pan(x1 - (15 - 3.4) * cf.MM, 0.5))  # round part's centre 15 mm from the frame
    grips, step = [], prim.env.step
    prim.env.step = lambda a: (grips.append(float(u._target_pos[4] + a[4])), step(a))[1]
    assert prim.grasp("pan") and prim.env.held == ["pan"]
    tips = np.concatenate(cf.finger_polys(u._target_pos)).mean(0)
    assert abs(max(grips) - WIDE_OPEN) < 1e-6 and np.linalg.norm(tips - (x1 - 15 * cf.MM, 0.5)) < 1.5e-3
    assert tuple(u._gripper_range) == (prim.closed, prim.open) and prim.open < WIDE_OPEN
    assert prim.place((0.45, 0.5)) and prim.env.held == []


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


def test_slot_is_carried_across_its_width(prim):
    x0 = cf.GLASS[0]
    # an end against the low-x frame: nothing reaches behind it, but the tips fit either side of its 15 mm width
    prim.scene.update(slot=_obj(x0 + 14 * cf.MM, 0.5, 25.5, 16, 15))
    assert not prim.push("slot", (1, 0), 30)
    assert prim.reset_objects() == {"slot": True} and any("grasp slot: close" in l for l, _ in prim.log)
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
    """Kinds by PLAY_WEIGHTS among those the scene allows; targets keep clear of the frame."""
    rng = np.random.default_rng(0)
    prim.scene.update(cube_blue=_obj(0.4, 0.4, 15, 15), cube_white=_obj(0.5, 0.6, 15, 15), slot=_obj(0.6, 0.4, 15, 16, 25.5),
                      pan=_obj(0.3, 0.6, 20, 7), rope=_obj(0.45, 0.25, 60, 8, 8))
    assert prim.goto("test: a step", x=0.4)  # the fixture wants at least one step sent
    picks = [prim.random_primitive(rng) for _ in range(400)]
    kinds = {k: sum(p[0] == k for p in picks) for k in PLAY_WEIGHTS}
    # cube into slot (weight 2) is always possible here; pick-and-place only when a free spot is drawn (FREE_MM)
    assert kinds["cube_out_of_slot"] == 0 and kinds["cube_into_slot"] > 1.5 * kinds["rope_drag"] and kinds["pick_place"] > 0
    x0, y0, x1, y1 = cf.GLASS
    for k, a in picks:
        if k == "pick_place":
            assert a[0] in PLAY_CARRY and x0 + 0.1 < a[1][0] < x1 - 0.1 and y0 + 0.1 < a[1][1] < y1 - 0.1
    prim.scene["cube_blue"] = SimpleNamespace(status="stacked", on="slot", in_notch=True, height_mm=15, inferred_xy=None)
    picks = [prim.random_primitive(rng) for _ in range(200)]
    assert ("cube_out_of_slot", ("cube_blue", "push")) in picks and not any(k == "cube_into_slot" for k, _ in picks)
    prim.scene["pan"] = _obj(0.08, 0.6, 20, 7)  # by the frame: never pushed further towards it
    assert not any(k == "push" and a[0] == "pan" and a[1][0] < 0 for k, a in (prim.random_primitive(rng) for _ in range(200)))


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
