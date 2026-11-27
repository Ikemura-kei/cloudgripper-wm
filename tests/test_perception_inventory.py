"""Naming logic of the perception module, on shape and colour numbers measured on cell 23 (2026-11-23). No GPU."""
import importlib.util

import numpy as np
import sys
from pathlib import Path

import pytest

# load the file directly: importing the package would pull in torch
_spec = importlib.util.spec_from_file_location(
    "scene_tracker", Path(__file__).resolve().parents[1] / "cloudgripper_wm/perception/scene_tracker.py")
st = importlib.util.module_from_spec(_spec)
sys.modules["scene_tracker"] = st
try:
    _spec.loader.exec_module(st)
except ImportError:  # torch or cv2 missing
    st = None
pytestmark = pytest.mark.skipif(st is None, reason="perception dependencies not installed")


def _cand(short, long, solidity, hsv, pieces=1):
    return dict(short=short, long=long, solidity=solidity, hsv=hsv, pieces=pieces)


def test_assign_inventory():
    cands = [
        _cand(52, 88, 0.96, (2, 168, 110)),     # slot
        _cand(69, 99, 0.90, (84, 87, 47)),      # pan
        _cand(37, 37, 0.97, (110, 110, 57)),    # blue cube
        _cand(34, 38, 0.98, (80, 14, 94)),      # white cube
        _cand(37, 37, 0.97, (75, 11, 50)),      # grey robot block of cube size: must be rejected
        _cand(105, 216, 0.34, (128, 39, 104)),  # rope: fits no rigid template
        _cand(34, 36, 0.96, (110, 110, 57), pieces=2),  # broken mask
    ]
    everything = {"cube_blue", "cube_white", "slot", "pan"}
    assert st.assign_inventory(cands, everything) == {"slot": 0, "pan": 1, "cube_blue": 2, "cube_white": 3}
    # only what is asked for is assigned, and an object that is not there stays unassigned
    assert st.assign_inventory(cands, {"cube_white"}) == {"cube_white": 3}
    assert st.assign_inventory(cands[:2], everything) == {"slot": 0, "pan": 1}
    # of two candidates for one name, the closer fit wins
    assert st.assign_inventory([_cand(40, 41, 0.96, (110, 110, 57)), cands[2]], everything) == {"cube_blue": 1}
    # a slot on its side fits no template (cell 23, 2026-11-24): the one red leftover of about its size is the slot
    side = _cand(65, 87, 0.91, (2, 177, 125))
    assert st.assign_inventory([cands[1], side, cands[2], cands[3]], everything) == {"pan": 0, "slot": 1, "cube_blue": 2, "cube_white": 3}
    assert st.assign_inventory([side, side], everything) == {}                               # two such leftovers: no guess
    assert st.assign_inventory([_cand(20, 30, 0.9, (2, 177, 125))], everything) == {}        # red but far too small


def test_top_cube_name():
    assert st._top_cube_name((109, 240, 142)) == "cube_blue"
    assert st._top_cube_name((44, 31, 216)) == "cube_white"
    assert st._top_cube_name((1, 201, 236)) is None    # the slot's red
    assert st._top_cube_name((114, 218, 100)) is None  # the blue frame in shade
    assert st._top_cube_name((110, 237, 110), in_container=True) == "cube_blue"  # tilted in the slot, in shade


def test_rope_one_piece():
    m = np.zeros((100, 100), bool)
    m[10:20, 10:80] = True   # the rope, 700 px
    m[60:64, 60:64] = True   # a patch on the rail
    one = st.one_piece(m, 1000)
    assert one.sum() == 700 and not one[60:64, 60:64].any()
    assert not st.one_piece(m, 500).any()                          # larger than a rope can be: not the rope
    assert not st.one_piece(np.zeros((5, 5), bool), 10).any()


def test_cube_hidden_from_below_keeps_its_place():
    """prim_22 (2026-11-26): the open tips over the white cube hid it from below; it became "off_glass" (a 30 mm,
    16 mm tall obstacle) and the grasp was refused. Unmoved in the top view, it stays where it was."""
    from types import SimpleNamespace
    tr = st.SceneTracker.__new__(st.SceneTracker)
    tr.base, tr.top = SimpleNamespace(masks={}), SimpleNamespace(masks={})
    tr._top_sat, tr._top_hsv, tr.H = None, np.zeros((720, 1280, 3), np.uint8), np.eye(3)
    tr._px_to_xy = lambda pts: np.asarray(pts, float).reshape(-1, 2) / 1000
    tr.states = {n: st.ObjectState(n, "unaccounted") for n in st.INVENTORY}

    def sq(shape, y, x, k):
        m = np.zeros(shape, bool)
        m[y:y + k, x:x + k] = True
        return m

    def see(base_y, top_y):
        tr.base.masks["cube_white"] = sq((480, 640), 200, 300, 32) if base_y else np.zeros((480, 640), bool)
        tr.top.masks["cube_white"] = sq((720, 1280), top_y, 600, 40)
        return tr._update()["cube_white"]
    s0 = see(True, 400)
    assert s0.status == "on_glass"
    s1 = see(False, 404)  # hidden from below, top view about the same: kept, same footprint
    assert s1.status == "on_glass" and np.allclose(s1.footprint_xy, s0.footprint_xy)
    # the fingers cover most of its top (play_1 episode 3): the visible part moved 14 px but lies where it was: kept
    tr.top.masks["cube_white"] = sq((720, 1280), 400, 600, 40) & (np.arange(1280) < 612)[None, :]
    assert tr._update()["cube_white"].status == "on_glass"
    assert see(False, 410).status == "on_glass"
    assert see(False, 416).status != "on_glass"  # crept 16 px from where it was last seen from below: no longer trusted
    assert see(False, 300).status != "on_glass"  # lifted


def test_cube_in_the_notch_hidden_from_above_stays():
    """play_1 episode 2 (2026-11-27): the arm over the slot hid the blue cube in its notch from the top camera, the only
    one that sees it there; it became "unaccounted" and the filter stopped the push. While the slot is seen from below
    it stays in the notch and moves with the slot; without the slot, nothing is assumed."""
    from types import SimpleNamespace
    tr = st.SceneTracker.__new__(st.SceneTracker)
    tr.base, tr.top = SimpleNamespace(masks={}), SimpleNamespace(masks={})
    tr._top_sat, tr._top_hsv, tr.H = None, np.zeros((720, 1280, 3), np.uint8), np.eye(3)
    tr._px_to_xy = lambda pts: np.asarray(pts, float).reshape(-1, 2) / 1000
    tr.states = {n: st.ObjectState(n, "unaccounted") for n in st.INVENTORY}

    def sq(shape, y, x, h, w=None):
        m = np.zeros(shape, bool)
        m[y:y + h, x:x + (w or h)] = True
        return m
    tr.base.masks["slot"] = sq((480, 640), 200, 300, 56, 86)
    tr.top.masks.update(slot=sq((720, 1280), 360, 560, 80, 110), cube_blue=sq((720, 1280), 380, 590, 40))
    s0 = tr._update()["cube_blue"]
    assert s0.status == "stacked" and s0.on == "slot" and s0.in_notch
    tr.base.masks["slot"] = sq((480, 640), 200, 300, 56, 86)
    tr.top.masks["cube_blue"] = np.zeros((720, 1280), bool)  # the arm over the slot
    s1 = tr._update()["cube_blue"]
    assert s1.status == "stacked" and s1.in_notch and s1.inferred_xy == s0.inferred_xy
    # one picture puts it elsewhere (the arm near the slot): kept, in doubt; a second one in a row is believed
    tr.top.masks["cube_blue"] = sq((720, 1280), 600, 900, 40)
    s2 = tr._update()["cube_blue"]
    assert s2.status == "stacked" and s2.in_notch and s2.doubt
    assert not tr._update()["cube_blue"].in_notch
    # back in the notch, then hidden while the slot is pushed (play_1 episode 4): it moves with the slot
    tr.top.masks["cube_blue"] = sq((720, 1280), 380, 590, 40)
    assert tr._update()["cube_blue"].in_notch
    tr.top.masks["cube_blue"] = np.zeros((720, 1280), bool)
    tr.base.masks["slot"] = sq((480, 640), 260, 300, 56, 86)
    s3 = tr._update()["cube_blue"]
    assert s3.status == "stacked" and s3.in_notch and s3.inferred_xy[1] > s0.inferred_xy[1] + 0.05  # 60 px on
    tr.base.masks["slot"] = np.zeros((480, 640), bool)  # the slot not seen either: nothing is assumed
    assert tr._update()["cube_blue"].status == "unaccounted"
