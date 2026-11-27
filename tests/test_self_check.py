"""Self-check (cloudgripper_wm/self_check.py): the measures on drawn pictures, then record and check on the mock robot."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cloudgripper_wm import collision_filter as cf
from cloudgripper_wm.envs.cloudgripper_env import CloudGripperEnv
from cloudgripper_wm.envs.constants import PLAY_MAX_DELTA
from cloudgripper_wm.envs.filtered_wrapper import FilteredCloudGripperWrapper
from cloudgripper_wm.envs.robot_pool import RobotPool
from cloudgripper_wm.perception.scene_tracker import load_states, save_states
from cloudgripper_wm.primitives import Primitives
from cloudgripper_wm.self_check import FRAME_SHIFT_PX, LOW_Z, NEAR_MIN, NEAR_PX, R, TILT_DEG, _frame, _measure, _near, _shift, self_check

YELLOW, NAVY = (0, 220, 240), (110, 40, 30)  # BGR


def _pics(tilt=0.0, gap=170, moved=0, rng=np.random.default_rng(0)):
    """Top view: the gripper's block (tilted by `tilt` deg), two fingers `gap` px apart, the cable's yellow stripe.
    Base view: the navy frame in two parts and the gripper, all `moved` px to the right (a camera that was moved).
    A little noise, so that no two pictures are the same array."""
    top = np.full((720, 1280, 3), 200, np.uint8)
    cv2.fillPoly(top, [cv2.boxPoints(((660, 40), (210, 80), tilt)).astype(np.int32)], YELLOW)
    for x in (660 - gap // 2, 660 + gap // 2):
        cv2.rectangle(top, (x - 15, 180), (x + 15, 340), YELLOW, -1)
    cv2.line(top, (760, 220), (1100, 300), YELLOW, 3)
    base = np.full((480, 640, 3), 200, np.uint8)
    cv2.rectangle(base, (60, 40), (580, 440), NAVY, 30)
    base[238:242] = 200  # the gap where the frame's two parts meet
    cv2.circle(base, (320, 240), 12, YELLOW, -1)
    base = np.roll(base, moved, 1)
    noise = lambda im: np.clip(im.astype(int) + rng.integers(-3, 4, im.shape), 0, 255).astype(np.uint8)
    return noise(top), noise(base)


def test_measures():
    m = _measure(*_pics())
    assert not m["claw"][:, 900:].any() and m["claw"].sum() > 2 * 30 * 150  # cable stripe gone, both fingers kept
    assert abs(m["tilt"]) < 1
    assert _near(m["claw"], _measure(*_pics(gap=176))["claw"], NEAR_PX) >= NEAR_MIN  # closed again, a few px off
    assert _near(m["claw"], _measure(*_pics(gap=60))["claw"], NEAR_PX) < NEAR_MIN  # did not open
    assert abs(_measure(*_pics(tilt=30))["tilt"]) > TILT_DEG
    f = _frame(_pics()[1])
    assert abs(_shift(f, _frame(_pics(moved=3)[1])) - 3) < 0.2  # a moved camera reads its shift
    f2 = f.copy(); f2[300:350, 40:95] = True  # the blue cube from below, against the frame
    assert _shift(f, f2) < 0.5


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
        top, base = _pics(tilt=30) if look.pop("glitch", False) else look["pics"]()
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
    assert status == "recorded" and (ref / "down_closed_top.jpg").exists() and (ref / "poses.json").exists()
    kept = (ref / "open_top.jpg").read_bytes()
    prim.look["pics"] = lambda: _pics(tilt=30)
    status, lines = self_check(prim, ref, record=True)  # recording again adds nothing and replaces nothing
    assert status == "recorded" and "0 new poses" in lines[0] and (ref / "open_top.jpg").read_bytes() == kept
    prim.look["pics"] = _pics
    status, lines = self_check(prim, ref, out=tmp_path / "out")
    assert status == "pass", "\n".join(lines)
    o = tmp_path / "out"
    assert len(list(o.glob("*.jpg"))) == 13 and any(l.startswith("down_open") for l in lines)
    log = [json.loads(l) for l in (o / "log.jsonl").read_text().splitlines()]
    assert len(log) == 13 and len(list((o / "raw").glob("*.jpg"))) == 26 and log[3]["label"] == "x+" and log[3]["claw"] == 1.0
    assert (o / "scene_start.npz").exists() and "PASS" in (o / "report.txt").read_text()
    assert prim.u._target_pos[2] == 1.0  # ends at the top height

    prim.look["pics"] = lambda: _pics(tilt=30)  # the gripper hangs tilted
    status, lines = self_check(prim, ref)
    assert status == "fail" and any("block edge" in l for l in lines)
    frozen = _pics()
    prim.look["pics"] = lambda: frozen  # the camera delivers the same frame
    status, lines = self_check(prim, ref)
    assert status == "fail" and any("same picture" in l for l in lines)
    prim.look["pics"] = lambda: _pics(moved=8)
    status, lines = self_check(prim, ref)
    assert status == "fail" and any("base camera moved" in l for l in lines) and FRAME_SHIFT_PX < 8

    prim.look["pics"] = _pics  # one bad picture at x+ (a glitch): pictured again, and the check passes
    goto = prim.goto
    prim.goto = lambda label, *a, **k: (goto(label, *a, **k), prim.look.update(glitch=label.endswith("x+")))[0]
    status, lines = self_check(prim, ref, out=tmp_path / "out2")
    assert status == "pass" and any(l.startswith("x+") and "pictured 2 times" in l for l in lines), "\n".join(lines)
    log = [json.loads(l) for l in (tmp_path / "out2/log.jsonl").read_text().splitlines()]
    assert [(d["take"], bool(d["bad"])) for d in log if d["label"] == "x+"] == [(0, True), (1, False)]  # both kept
    prim.goto = goto

    prim.look["pics"] = _pics
    tips = np.concatenate(cf.finger_polys([0.5, 0.5, LOW_Z, R, prim.open])).mean(0)
    a = 12 * cf.MM  # under the gripper, as tall as the slot with a cube in it: no height check, the rest runs
    prim.scene["slot"] = SimpleNamespace(status="on_glass", on=None, xy=tuple(tips), height_mm=18.5, inferred_xy=None,
                                         footprint_xy=tips + np.array([[-a, -a], [a, -a], [a, a], [-a, a]]))
    status, lines = self_check(prim, ref)
    assert status == "pass" and "height not checked" in lines[0] and not any(l.startswith("down_") for l in lines), "\n".join(lines)
    assert self_check(prim, tmp_path / "ref2", record=True)[0] == "blocked"  # a reference is recorded whole


def test_states_roundtrip(tmp_path):
    m = np.zeros((480, 640), bool); m[100:150, 200:260] = True
    s = SimpleNamespace(name="cube_blue", status="on_glass", xy=(0.4, 0.5), footprint_xy=np.ones((4, 2)), mask_base=m, mask_top=None)
    save_states(tmp_path / "s.npz", {"cube_blue": s})
    back = load_states(tmp_path / "s.npz")["cube_blue"]
    assert (back["mask_base"] == m).all() and back["footprint_xy"] == [[1.0, 1.0]] * 4 and back["status"] == "on_glass"
    assert (tmp_path / "s.npz").stat().st_size < 2000
