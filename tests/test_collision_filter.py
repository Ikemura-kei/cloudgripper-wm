"""Collision filter rules on made-up scenes (cell 23 geometry). No GPU, no robot."""
from types import SimpleNamespace

import numpy as np

from cloudgripper_wm import collision_filter as cf

MM = cf.MM


def _obj(cx, cy, w_mm, h_mm, status="on_glass", on=None):
    a, b = w_mm / 2 * MM, w_mm / 2 * MM
    return SimpleNamespace(status=status, on=on, height_mm=h_mm,
                           footprint_xy=np.array([[cx - a, cy - b], [cx + a, cy - b], [cx + a, cy + b], [cx - a, cy + b]]))


def _pose(fx, fy, z=0.45):
    """Commanded pose that puts the fingers' midpoint at fx, fy."""
    return [fx - cf.TOOL_OFFSET[0], fy - cf.TOOL_OFFSET[1], z, 0.0, 1.0]


def test_rules():
    step = 0.025
    slot = dict(slot=_obj(0.5, 0.5, 20, 16))
    # nothing near: free
    assert cf.check(_pose(0.3, 0.3), _pose(0.3 + step, 0.3), slot).reason == "free"
    # into a free 16 mm object mid-glass: a push, allowed
    v = cf.check(_pose(0.5, 0.36), _pose(0.5, 0.36 + step), slot)
    assert v.allowed and v.pushed == ["slot"]
    # over a 10 mm cube at tip height 14 mm (z 0.45): no contact
    fx = 0.5 + cf.FINGER_SEP_MM / 2 * MM  # puts one finger in line with the cube (centred, the open fingers straddle it)
    assert cf.check(_pose(fx, 0.40), _pose(fx, 0.40 + step), dict(cube_blue=_obj(0.5, 0.5, 10, 10))).reason == "free"
    # ... but a cube stacked on the pan makes the pan 17 mm tall
    stack = dict(pan=_obj(0.5, 0.5, 20, 7), cube_blue=SimpleNamespace(status="stacked", on="pan", height_mm=10, footprint_xy=None))
    assert cf.check(_pose(0.5, 0.36), _pose(0.5, 0.36 + step), stack).pushed == ["pan"]

    # the same push with the slot against the frame (y1 = 0.873): blocked; away from the frame: allowed
    wall = dict(slot=_obj(0.5, 0.79, 20, 16))
    assert not cf.check(_pose(0.5, 0.65), _pose(0.5, 0.65 + step), wall).allowed
    v = cf.check(_pose(0.36, 0.79), _pose(0.36 + step, 0.79), wall)  # along the frame
    assert v.allowed and v.pushed == ["slot"], v.reason
    # a chain: the pushed slot touches a cube that is against the frame
    chain = dict(slot=_obj(0.5, 0.63, 20, 16), cube_white=_obj(0.5, 0.80, 10, 10))
    chain["slot"].footprint_xy[:, 1] += np.array([0, 0, 0.09, 0.09])  # stretch it to reach the cube
    v = cf.check(_pose(0.5, 0.49), _pose(0.5, 0.49 + step), chain)
    assert not v.allowed and "cube_white" in v.reason, v.reason

    # at z 0.25 (10.5 mm) the tips no longer clear a cube: travelling into it is a push
    assert cf.check(_pose(fx, 0.40, z=0.25), _pose(fx, 0.40 + step, z=0.25), dict(cube_blue=_obj(0.5, 0.5, 10, 10))).pushed == ["cube_blue"]
    # an unaccounted object blocks everything
    lost = dict(cube_blue=SimpleNamespace(status="unaccounted", on=None, height_mm=10, footprint_xy=None))
    assert not cf.check(_pose(0.3, 0.3), _pose(0.3 + step, 0.3), lost).allowed


def test_heights():
    assert [cf.tip_mm(z) for z in (0.0, 0.5, 1.0)] == [6.0, 15.0, 22.0] and abs(cf.tip_mm(0.45) - 14.1) < 1e-9


def test_lowering():
    cube = dict(cube_blue=_obj(0.5, 0.5, 10, 10))
    fx = 0.5 + cf.FINGER_SEP_MM / 2 * MM  # fingers' midpoint that puts one finger over the cube
    assert not cf.check(_pose(fx, 0.5), _pose(fx, 0.5, z=0.0), cube).allowed
    # the open fingers go down around a cube: centred, at any angle
    assert cf.check(_pose(0.5, 0.5), _pose(0.5, 0.5, z=0.0), cube).allowed
    turned = dict(cube_blue=_obj(0.5, 0.5, 10, 10))
    c, r = np.array([0.5, 0.5]), np.sqrt(0.5) * np.array([[1, -1], [1, 1]])
    turned["cube_blue"].footprint_xy = (turned["cube_blue"].footprint_xy - c) @ r.T + c  # 45 deg
    assert cf.check(_pose(0.5, 0.5), _pose(0.5, 0.5, z=0.0), turned).allowed
    # ... 2 mm off centre along the gripper axis still fits, 3.5 mm does not
    assert cf.check(_pose(0.5 + 2 * MM, 0.5), _pose(0.5 + 2 * MM, 0.5, z=0.0), cube).allowed
    assert not cf.check(_pose(0.5 + 3.5 * MM, 0.5), _pose(0.5 + 3.5 * MM, 0.5, z=0.0), cube).allowed
    # ... and once down, moving sideways pushes the cube, which is allowed
    v = cf.check(_pose(0.5, 0.5, z=0.0), _pose(0.5 + 0.025, 0.5, z=0.0), cube)
    assert v.allowed and v.pushed == ["cube_blue"]
    assert cf.check(_pose(0.3, 0.3, z=0.0), _pose(0.3, 0.3), cube).allowed  # going up
    # lowering beside everything is free, and so is lowering a little while still clear of the cube below
    assert cf.check(_pose(0.3, 0.3), _pose(0.3, 0.3, z=0.0), cube).allowed
    assert cf.check(_pose(fx, 0.5, z=1.0), _pose(fx, 0.5, z=0.6), cube).allowed  # 16.4 mm over a 10 mm cube


def test_cell20_gripper_and_stray_cube():
    """Cell 20: the gripper command sets the tip spacing; opening is checked, closing is a grasp."""
    keep = {k: getattr(cf, k) for k in list(cf.CELL20) + ["_FRAME"]}
    cf.use_cell("robot20")
    try:
        x0, y0, x1, y1 = cf.GLASS
        at = lambda fx, fy, g, z=0.0: [fx - cf.TOOL_OFFSET[0] - cf.TOOL_OFFSET_TURNING[0], fy - cf.TOOL_OFFSET[1] - cf.TOOL_OFFSET_TURNING[1], z, 0.0, g]
        cube = lambda cx, cy: dict(cube_blue=_obj(cx, cy, 10, 10))
        # mid-glass: opening and closing in the air and beside a cube are free; closing around it is a grasp
        assert cf.check(at(0.5, 0.5, 0.1), at(0.5, 0.5, 0.825), cube(0.2, 0.2)).reason == "free"
        v = cf.check(at(0.5, 0.5, 0.825), at(0.5, 0.5, 0.1), cube(0.5, 0.5))
        assert v.allowed and v.pushed == ["cube_blue"]
        # opening with a cube just outside one tip pushes it: allowed mid-glass ...
        out = 0.5 + (cf.sep_mm(0.1) / 2 + cf.FINGER_MM[0] / 2 + 6) * MM
        v = cf.check(at(0.5, 0.5, 0.1), at(0.5, 0.5, 0.825), cube(out, 0.5))
        assert v.allowed and v.pushed == ["cube_blue"], v.reason
        # ... blocked when the frame is right behind the cube, and when the tip itself would open into the frame
        fx = x1 - 22 * MM
        v = cf.check(at(fx, 0.5, 0.1), at(fx, 0.5, 0.825), cube(x1 - 6 * MM, 0.5))
        assert not v.allowed and "opening" in v.reason, v.reason
        fx = x1 - 9 * MM
        assert "opening against the frame" in cf.check(at(fx, 0.5, 0.1), at(fx, 0.5, 0.825), {}).reason
        assert cf.check(at(fx, 0.5, 0.1, z=1.0), at(fx, 0.5, 0.825, z=1.0), {}).allowed  # above the frame: free
        # a normal grasp, start to end: down around the cube, close, up, down again holding it, sideways, let go
        c, g = cube(0.5, 0.5), (0.825, 0.1)
        steps = [(at(0.5, 0.5, g[0], 1.0), at(0.5, 0.5, g[0], 0.0)), (at(0.5, 0.5, g[0]), at(0.5, 0.5, g[1])),
                 (at(0.5, 0.5, g[1]), at(0.5, 0.5, g[1], 0.3)), (at(0.5, 0.5, g[1], 0.3), at(0.5, 0.5, g[1])),
                 (at(0.5, 0.5, g[1]), at(0.525, 0.5, g[1])), (at(0.5, 0.5, g[1]), at(0.5, 0.5, g[0]))]
        assert [cf.check(p0, p1, c).allowed for p0, p1 in steps] == [True] * 6, [cf.check(p0, p1, c).reason for p0, p1 in steps]
        # ... and letting go of it is allowed right beside the frame, where opening onto a free cube is not
        fx = x1 - 22 * MM
        assert cf.check(at(fx, 0.5, 0.1), at(fx, 0.5, 0.825), cube(fx, 0.5)).allowed
        # closed tips over a cube that is NOT between them (under one tip only): no lowering
        off = 0.5 + (cf.sep_mm(0.1) / 2 + 7) * MM
        assert not cf.check(at(0.5, 0.5, 0.1, 1.0), at(0.5, 0.5, 0.1), cube(off, 0.5)).allowed
        # a cube off the glass (seen only from above) is an obstacle where the top view puts it
        stray = dict(cube_white=SimpleNamespace(status="off_glass", on=None, height_mm=10, footprint_xy=None, inferred_xy=(0.5, 0.5)))
        assert not cf.check(at(0.5, 0.5, 0.1, z=1.0), at(0.5, 0.5, 0.1, z=0.3), stray).allowed
        assert cf.check(at(0.2, 0.2, 0.1, z=1.0), at(0.2, 0.2, 0.1, z=0.3), stray).allowed
        # a cube "stacked on the pan" may lean beside it: no lowering within a cube of the pan
        lean = dict(pan=_obj(0.5, 0.5, 20, 7), cube_blue=SimpleNamespace(status="stacked", on="pan", height_mm=10, footprint_xy=None))
        beside = 0.5 + (10 + 8) * MM
        assert not cf.check(at(0.5, beside, 0.1, z=1.0), at(0.5, beside, 0.1, z=0.0), lean).allowed
    finally:
        for k, val in keep.items():
            setattr(cf, k, val)
