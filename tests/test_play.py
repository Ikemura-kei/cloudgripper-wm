"""Play collection (cloudgripper_wm/play.py) on the mock robot: episodes and their files, the conversion to Lance, a
surprise that brings a self-check, the stops."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cloudgripper_wm import collision_filter as cf
from cloudgripper_wm.envs.cloudgripper_env import CloudGripperEnv
from cloudgripper_wm.envs.constants import PLAY_MAX_DELTA
from cloudgripper_wm.envs.filtered_wrapper import FilteredCloudGripperWrapper
from cloudgripper_wm.envs.robot_pool import RobotPool
from cloudgripper_wm.play import Recorder, collect
from cloudgripper_wm.primitives import Primitives


def _obj(x, y, w_mm, h_mm, l_mm=None):
    a, b = w_mm / 2 * cf.MM, (l_mm or w_mm) / 2 * cf.MM
    return SimpleNamespace(status="on_glass", on=None, xy=(x, y), height_mm=h_mm, inferred_xy=None, mask_base=None, mask_top=None,
                           footprint_xy=np.array([[x - a, y - b], [x + a, y - b], [x + a, y + b], [x - a, y + b]]))


@pytest.fixture
def setup(tmp_path):
    keep = {k: getattr(cf, k) for k in list(cf.CELL20) + ["_FRAME"]}
    scene = dict(cube_blue=_obj(0.4, 0.4, 15, 15), cube_white=_obj(0.5, 0.6, 15, 15), slot=_obj(0.6, 0.4, 15, 16, 25.5),
                 pan=_obj(0.3, 0.6, 20, 7), rope=_obj(0.45, 0.25, 60, 8, 8))
    tracker = SimpleNamespace(start=lambda b, t: scene, step=lambda b, t: scene, refresh=lambda b, t: scene)
    RobotPool.configure(["robot20"])
    env = FilteredCloudGripperWrapper(CloudGripperEnv(use_mock=True, dwell_time=0, reset_dwell_time=0, max_delta=PLAY_MAX_DELTA), tracker=tracker)
    env.reset()
    rec = Recorder(env, tmp_path)
    prim = Primitives(env, on_step=rec.step)
    yield prim, rec, scene, tmp_path
    env.close()
    for k, val in keep.items():
        setattr(cf, k, val)


def test_collect(setup):
    prim, rec, scene, out = setup
    calls = []
    def check(p, ref=None, out=None, record=False):  # self_check's signature: a folder passed as `ref` would be a bug
        assert ref is None and out is not None
        Path(out).mkdir(parents=True)
        calls.append(Path(out).name)
        return "pass", ["ok"]
    assert collect(prim, rec, out, 2, np.random.default_rng(1), check=check) == "done" and calls == ["000_a"]
    eps = sorted((out / "episodes").glob("ep_*"))
    assert [e.name for e in eps] == ["ep_00000", "ep_00001"]
    frames = 0
    for ep in eps:
        z = np.load(ep / "steps.npz")
        n = len(z["action"])
        assert 0 < n and z["state"].shape == (n + 1, 5) and z["action"].shape == (n, 5)
        assert len(list((ep / "frames").glob("*_top.jpg"))) == n + 1 == len(list((ep / "states").glob("*.npz")))
        diag = [json.loads(l) for l in (ep / "diag.jsonl").read_text().splitlines()]
        assert len(diag) == n and all(d["verdict"] and len(d["executed"]) == 5 for d in diag)
        frames += n + 1
    events = [json.loads(l)["event"] for l in (out / "session.jsonl").read_text().splitlines()]
    assert events.count("episode") == 2 and events.count("episode_end") == 2 and "stop" not in events


    # a surprise brings a self-check before the next episode
    push = prim.push
    prim.random_primitive = lambda rng: ("push", ("cube_blue", (1.0, 0.0), 10.0))
    prim.push = lambda *a: (setattr(prim, "surprise", "test"), push(*a))[1]
    calls.clear()
    assert collect(prim, rec, out, 2, np.random.default_rng(2), check=check) == "done" and calls == ["001_a", "002_a"]
    # a self-check that fails twice stops, arm up and gripper open
    assert collect(prim, rec, out, 1, np.random.default_rng(3), check=lambda p, ref=None, out=None: ("fail", ["bad"])).startswith("self-check failed twice")
    assert prim.u._target_pos[2] == 1.0 and np.isclose(prim.u._target_pos[4], prim.open)
    # a scene the reset cannot fix stops before the episode
    scene["pan"] = SimpleNamespace(status="unaccounted", on=None, height_mm=7, inferred_xy=None, footprint_xy=None)
    n = len(list((out / "episodes").glob("ep_*")))
    assert collect(prim, rec, out, 1, np.random.default_rng(4), check=check).startswith("scene not fit after a reset")
    assert len(list((out / "episodes").glob("ep_*"))) == n


def test_play_to_lance(tmp_path):
    """Frames and steps.npz to a Lance dataset; an episode without steps.npz (interrupted) is skipped."""
    pytest.importorskip("lancedb")  # stable-worldmodel's data extra: in the script's own uv environment, not the parent venv
    import cv2
    for e, n in ((0, 3), (1, 2), (2, 0)):
        d = tmp_path / "episodes" / f"ep_{e:05d}" / "frames"
        d.mkdir(parents=True)
        for i in range(n + 1):
            cv2.imwrite(str(d / f"{i:04d}_top.jpg"), np.full((72, 128, 3), 40 * i, np.uint8))
            cv2.imwrite(str(d / f"{i:04d}_base.jpg"), np.full((48, 64, 3), 40 * i, np.uint8))
        if e < 2:
            np.savez(d.parent / "steps.npz", state=np.zeros((n + 1, 5), np.float32), action=np.ones((n, 5), np.float32))
    spec = importlib.util.spec_from_file_location("p2l", Path(__file__).parents[1] / "scripts/data/play_to_lance.py")
    p2l = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p2l)
    p2l.convert(str(tmp_path / "train.lance"), [tmp_path])
    import lance
    t = lance.dataset(str(tmp_path / "train.lance")).to_table()
    assert t.num_rows == 4 + 3 and {"pixels", "pixels_base", "state", "action"} <= set(t.column_names)


def test_collect_uses_the_real_self_check_signature(setup, monkeypatch):
    """The default check is self_check.self_check: called with the output folder by name, against the reference in git."""
    prim, rec, scene, out = setup
    import cloudgripper_wm.self_check as sc
    seen = {}
    monkeypatch.setattr(sc, "self_check", lambda p, ref=None, out=None, record=False: (seen.update(ref=ref, out=out), ("fail", ["x"]))[1])
    collect(prim, rec, out, 1, np.random.default_rng(0))
    assert seen["ref"] is None and str(seen["out"]).endswith("self_checks/000_b")
