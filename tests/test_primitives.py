"""Play primitives on the mock robot with a made-up scene (cell 20): the sequences run, and a blocked move gives up."""
from types import SimpleNamespace

import numpy as np
import pytest

from cloudgripper_wm import collision_filter as cf
from cloudgripper_wm.envs.cloudgripper_env import CloudGripperEnv
from cloudgripper_wm.envs.constants import PLAY_MAX_DELTA
from cloudgripper_wm.envs.filtered_wrapper import FilteredCloudGripperWrapper
from cloudgripper_wm.envs.robot_pool import RobotPool
from cloudgripper_wm.primitives import Primitives, z_for_tip


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

