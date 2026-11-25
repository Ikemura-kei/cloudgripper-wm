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


def test_rope_one_piece():
    m = np.zeros((100, 100), bool)
    m[10:20, 10:80] = True   # the rope, 700 px
    m[60:64, 60:64] = True   # a patch on the rail
    one = st.one_piece(m, 1000)
    assert one.sum() == 700 and not one[60:64, 60:64].any()
    assert not st.one_piece(m, 500).any()                          # larger than a rope can be: not the rope
    assert not st.one_piece(np.zeros((5, 5), bool), 10).any()
