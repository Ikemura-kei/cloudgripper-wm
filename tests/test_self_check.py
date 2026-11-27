"""Self-check (cloudgripper_wm/self_check.py): the measures on drawn pictures, then record and check on the mock robot."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cloudgripper_wm import collision_filter as cf
from cloudgripper_wm.envs.cloudgripper_env import CloudGripperEnv
from cloudgripper_wm.envs.constants import PLAY_MAX_DELTA
from cloudgripper_wm.envs.filtered_wrapper import FilteredCloudGripperWrapper
from cloudgripper_wm.envs.robot_pool import RobotPool
from cloudgripper_wm.primitives import Primitives
from cloudgripper_wm.self_check import LOW_Z, NEAR_MIN, NEAR_PX, R, TILT_DEG, _measure, _near, self_check

YELLOW = (0, 220, 240)  # BGR


def _pics(tilt=0.0, gap=170, rng=np.random.default_rng(0)):
    """Top view: the gripper's block (tilted by `tilt` deg), two fingers `gap` px apart, the cable's yellow stripe.
    Base view: the blue frame and the gripper. A little noise, so that no two pictures are the same array."""
    top = np.full((720, 1280, 3), 200, np.uint8)
    cv2.fillPoly(top, [cv2.boxPoints(((660, 40), (210, 80), tilt)).astype(np.int32)], YELLOW)
    for x in (660 - gap // 2, 660 + gap // 2):
        cv2.rectangle(top, (x - 15, 180), (x + 15, 340), YELLOW, -1)
    cv2.line(top, (760, 220), (1100, 300), YELLOW, 3)
    base = np.full((480, 640, 3), 200, np.uint8)
    cv2.rectangle(base, (60, 40), (580, 440), (200, 40, 20), 30)
    cv2.circle(base, (320, 240), 12, YELLOW, -1)
    noise = lambda im: np.clip(im.astype(int) + rng.integers(-3, 4, im.shape), 0, 255).astype(np.uint8)
    return noise(top), noise(base)


def test_measures():
    m = _measure(*_pics())
    assert not m["claw"][:, 900:].any() and m["claw"].sum() > 2 * 30 * 150  # cable stripe gone, both fingers kept
    assert abs(m["tilt"]) < 1
    assert _near(m["claw"], _measure(*_pics(gap=176))["claw"], NEAR_PX) >= NEAR_MIN  # closed again, a few px off
    assert _near(m["claw"], _measure(*_pics(gap=60))["claw"], NEAR_PX) < NEAR_MIN  # did not open
    assert abs(_measure(*_pics(tilt=30))["tilt"]) > TILT_DEG


@pytest.fixture
def prim():
    keep = {k: getattr(cf, k) for k in list(cf.CELL20) + ["_FRAME"]}
    scene = {}
    tracker = SimpleNamespace(start=lambda b, t: scene, step=lambda b, t: scene, refresh=lambda b, t: scene)
    RobotPool.configure(["robot20"])
    env = FilteredCloudGripperWrapper(CloudGripperEnv(use_mock=True, dwell_time=0, reset_dwell_time=0, max_delta=PLAY_MAX_DELTA), tracker=tracker)
    env.reset()
    u, look = env.unwrapped, {"pics": _pics}

    def observe():
        top, base = look["pics"]()
        return top, cv2.cvtColor(base, cv2.COLOR_BGR2RGB), u._target_pos.copy()
    u._observe = observe
    p = Primitives(env)
    p.scene, p.look = scene, look
    yield p
    env.close()
    for k, val in keep.items():
        setattr(cf, k, val)


def test_record_then_check(prim, tmp_path):
    ref = tmp_path / "ref"
    status, lines = self_check(prim, ref, record=True)
    assert status == "recorded" and (ref / "s2_low_open_top.jpg").exists() and (ref / "poses.json").exists()
    with pytest.raises(FileExistsError):
        self_check(prim, ref, record=True)
    status, lines = self_check(prim, ref, out=tmp_path / "out")
    assert status == "pass", "\n".join(lines)
    assert len(list((tmp_path / "out").glob("*.jpg"))) == 14 and any(l.startswith("s0_low_open") for l in lines)
    assert prim.u._target_pos[2] == 1.0  # ends at the top height

    prim.look["pics"] = lambda: _pics(tilt=30)  # the gripper hangs tilted
    status, lines = self_check(prim, ref)
    assert status == "fail" and any("block edge" in l for l in lines)
    frozen = _pics()
    prim.look["pics"] = lambda: frozen  # the camera delivers the same frame
    status, lines = self_check(prim, ref)
    assert status == "fail" and any("same picture" in l for l in lines)

    prim.look["pics"] = _pics
    tips = np.concatenate(cf.finger_polys([0.5, 0.5, LOW_Z, R, prim.open])).mean(0)
    a = 12 * cf.MM  # 24 mm, wider than the open tips (22.5 mm), under the first low spot: the next one is used
    prim.scene["cube_blue"] = SimpleNamespace(status="on_glass", on=None, xy=tuple(tips), height_mm=15, inferred_xy=None,
                                              footprint_xy=tips + np.array([[-a, -a], [a, -a], [a, a], [-a, a]]))
    status, lines = self_check(prim, ref)
    assert status == "pass" and any(l.startswith("s1_low_open") for l in lines) and not any(l.startswith("s0_") for l in lines), "\n".join(lines)
