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
    # a cube against either frame is picked up (tips along the frame, a little towards the glass) and put down
    for x in (x0 + 5 * cf.MM, x1 - 5 * cf.MM):
        prim.scene.clear(); prim.scene.update(cube_blue=_obj(x, 0.5, 10, 10))
        assert prim.reset_objects() == {"cube_blue": True}, prim.log[-3:]
        assert abs(prim.env.unwrapped._target_pos[3] - 0.5) < 1e-6  # tips along y, the frame's direction there


def test_grasp_place_slot_reset_rope(prim):
    prim.scene.update(cube_blue=_obj(0.3, 0.3, 10, 10), slot=_obj(0.6, 0.6, 15, 16, 25.5))
    assert prim.pick_place("cube_blue", (0.5, 0.3)) and prim.env.held == []
    assert prim.cube_into_slot("cube_blue") and prim.env.held == []
    let_go = [l for l, _ in prim.log].index("cube_blue into slot: let go")
    assert "down" in prim.log[let_go - 1][0] and not prim.log[let_go - 1][1]
    prim.scene.clear()
    prim.scene.update(cube_white=_obj(0.75, 0.75, 10, 10), pan=_obj(0.45, 0.5, 20, 7))  # only the cube is far from the centre
    assert prim.reset_objects() == {"cube_white": True}  # in a corner: no grasp fits, it is pushed out diagonally
    assert any("grasp cube_white" in l and b for l, b in prim.log) and prim.log[-1][0] == "push cube_white: up"
    rope = _obj(0.5, 0.2, 60, 8, 8); rope.footprint_xy = np.array([[x, 0.2 + s * 4 * cf.MM] for s in (-1, 1) for x in np.linspace(0.3, 0.7, 30)[::s]])
    prim.scene.clear(); prim.scene.update(rope=rope)
    assert prim.rope_drag(0, (0.0, 0.1)) and prim.env.held == []
