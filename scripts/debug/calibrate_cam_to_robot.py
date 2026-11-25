# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["numpy", "opencv-python-headless", "pyyaml", "scikit-learn", "requests", "websocket-client"]
# ///
"""Make a cell's pixel-to-robot file (camera_params/cam-to-robot-points/camera-to-robot-crN.yaml). THIS MOVES THE ROBOT.

  export CLOUDGRIPPER_TOKEN=...
  uv run scripts/debug/calibrate_cam_to_robot.py --robot robot20 --out DIR --look     # one picture, no motion
  uv run scripts/debug/calibrate_cam_to_robot.py --robot robot20 --out DIR            # the grid run, about 8 min
  uv run scripts/debug/calibrate_cam_to_robot.py --robot robot20 --out DIR --fit-only # redo detection + fit from DIR

The glass must be EMPTY (no frame, no objects): the arm travels at height --z (default 0, the lowest) the whole time.
It needs the cell's fisheye file (base-camera-calibration/camera-params-crN.yaml) first.

The gripper is closed, then the arm visits an --n x --n grid of commanded x, y within --lo .. --hi. At each point the
undistorted base view is taken at rotation 0 and at 180 degrees. The file maps commanded x, y to the pixel of the two
finger tips' MIDPOINT AT ROTATION 0, at height --z (Kei, 2026-11-25). The 180-degree pictures give the rotation axis
(halfway between the two detections), printed as an offset from that midpoint for the collision filter.
Positions where the picture's edge cuts the tips off are left out: check the printed list.

--target cube --prior YAML: a blue cube held in the gripper is dragged through the grid instead (rotation 0 only) and
its outline on the glass gives the centre, at glass level and without the fingers' bodies in the way. Close the
gripper on the cube first and pass that command as --grip. The prior (e.g. a finger run's yaml) only picks the right
blue patch.

DIR gets every picture, points.json, check.jpg (detected tips = circle, refit = cross) and the yaml, which is copied
into camera_params/ only with --install."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cloudgripper-api"), str(ROOT)]
from cloudgripper_wm.utils.coordinate_converter import CoordinateConverter  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--robot", required=True)
p.add_argument("--out", required=True, type=Path)
p.add_argument("--n", type=int, default=6)
p.add_argument("--lo", type=float, default=0.05)
p.add_argument("--hi", type=float, default=0.95)
p.add_argument("--xs", help="comma list of x values instead of the --n grid's (e.g. to add a row to an earlier run)")
p.add_argument("--ys", help="comma list of y values instead of the --n grid's")
p.add_argument("--z", type=float, default=0.0, help="height command the mapping is defined at")
p.add_argument("--grip", type=float, default=0.2, help="gripper command for closed")
p.add_argument("--target", choices=["fingers", "cube"], default="fingers",
               help="what marks the centre: the closed fingers' yellow, or a blue cube held in the gripper (needs --prior)")
p.add_argument("--prior", type=Path, help="cube: an earlier, rougher camera-to-robot yaml, to tell the cube from other blue")
p.add_argument("--look", action="store_true", help="save one picture and exit, no motion")
p.add_argument("--fit-only", action="store_true", help="no robot: detect and fit from the pictures in --out")
p.add_argument("--install", action="store_true", help="copy the yaml into camera_params/")
a = p.parse_args()
n = a.robot[5:]
cams = ROOT / "cloudgripper_wm/camera_params"
cal = yaml.safe_load(open(cams / f"base-camera-calibration/camera-params-cr{n}.yaml"))
K, D = np.array(cal["K"]), np.array(cal["D"])
a.out.mkdir(parents=True, exist_ok=True)


def undistort(img):  # as CloudGripperImageProcessor.undistort_fish_eye
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, img.shape[:2][::-1], cv2.CV_16SC2)
    return cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR)


prior = CoordinateConverter(str(a.prior)) if a.prior else None


def yellow(img, xy=None):
    """Pixel coordinates (xs, ys) of the target in an undistorted base picture taken at commanded xy: the fingers'
    yellow, or (--target cube) the blue patch nearest where the prior mapping puts the gripper."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if a.target == "cube":
        # ponytail: the arm's cable chain has the cube's blue; told apart by position only (nearest patch within 40 px)
        m = cv2.morphologyEx(cv2.inRange(hsv, (100, 90, 30), (125, 255, 255)), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        k, lab, st, cen = cv2.connectedComponentsWithStats(m)
        d = np.hypot(*(cen[1:] - prior.x_y_to_px_py(*xy)).T) if k > 1 else np.array([1e9])
        ys, xs = np.nonzero(lab == 1 + int(np.argmin(d))) if d.min() < 40 else (np.array([], int), np.array([], int))
        return xs, ys
    # ponytail: colour threshold for the yellow printed fingers; a cell with other yellow under the arm needs a mask
    m = cv2.inRange(hsv, (8, 90, 110), (35, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))  # drops the thin orange servo wire
    ys, xs = np.nonzero(m)
    return xs, ys


def fingers(img, size, xy=None):
    """Midpoint (px, py) of the two closed finger tips in an undistorted base picture, or None.
    size = (extent in px, extent in py) of the finger pair seen straight from below (measured at the image centre).
    The camera looks up, so off-centre it also sees the fingers' sides and the mount above them, all drawn from the
    tips TOWARD the image centre (higher = nearer the centre). Only the pair's outer edges are therefore at tip
    height: the midpoint is the yellow's edge on the side away from the centre, minus half the pair's extent."""
    xs, ys = yellow(img, xy)
    if len(xs) < 150:
        return None
    r = np.hypot(xs - K[0, 2], ys - K[1, 2])
    band = r >= np.percentile(r, 99) - (max(size) + 8)  # the tips' end of the yellow, about one pair deep
    out = []
    for v, c, ext, lim in ((xs[band], K[0, 2], size[0], img.shape[1]), (ys[band], K[1, 2], size[1], img.shape[0])):
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        if (hi if v.mean() > c else lo) in (0, lim - 1) or hi > lim - 3 and v.mean() > c or lo < 2 and v.mean() <= c:
            return None  # that edge is cut off by the picture's edge
        out.append(float(hi - ext / 2 if v.mean() > c else lo + ext / 2))
    return tuple(out)


xs, ys = ([float(v) for v in l.split(",")] if l else np.linspace(a.lo, a.hi, a.n) for l in (a.xs, a.ys))
grid = [(round(float(x), 4), round(float(y), 4)) for i, x in enumerate(xs) for y in (ys[::1 if i % 2 == 0 else -1])]  # there and back
assert all(0.05 <= v <= 0.95 for q in grid for v in q)

ROTS = (0, 180) if a.target == "fingers" else (0,)  # a held cube is its own centre: rotation 0 only
if not a.fit_only:
    from client.cloudgripper_client import GripperRobot  # noqa: E402
    robot = GripperRobot(a.robot, os.environ.get("CLOUDGRIPPER_TOKEN") or sys.exit("set CLOUDGRIPPER_TOKEN"))

    def shot(tag, wait=2.0):
        """Wait, then poll until two base frames 1 s apart agree; save the undistorted base and the top view."""
        time.sleep(wait)
        prev = None
        for _ in range(20):
            state, _, base, _, top, _ = robot.get_all_states()
            if state is None:
                sys.exit(f"getAllStates failed at {tag}; arm left where it is")
            g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(int)
            if prev is not None and (abs(g - prev) > 25).mean() < 0.002:
                break
            prev = g
            time.sleep(1)
        cv2.imwrite(str(a.out / f"{tag}_base.png"), undistort(base))
        cv2.imwrite(str(a.out / f"{tag}_top.jpg"), top)
        return state

    def do(fn, *args):
        if fn(*args) is None:
            sys.exit(f"{fn.__name__}{args} failed; arm left where it is")

    if a.look:
        print("look:", shot("look", 0), "->", a.out / "look_base.png")
        sys.exit()
    do(robot.move_gripper, a.grip)
    do(robot.rotate, 0)
    do(robot.move_z, a.z)
    for i, (x, y) in enumerate(grid):
        do(robot.move_xy, x, y)
        for rot in ROTS:
            do(robot.rotate, rot)
            s = shot(f"x{x}_y{y}_r{rot}", 3.0 if rot == 0 else 2.0)
        print(f"{i + 1}/{len(grid)}  x {x} y {y}  z current {s['z_current']}", flush=True)
    do(robot.rotate, 0)
    print(f"arm left at x {grid[-1][0]} y {grid[-1][1]} z {a.z} rotation 0, gripper {a.grip}")

# every position photographed into --out so far, this run's or an earlier one's
done = sorted({tuple(float(v[1:]) for v in f.name.split("_")[:2]) for f in a.out.glob("x*_y*_r0_base.png")})
# the finger pair's extent, from the rotation-0 picture where the yellow is nearest the image centre (no sides seen)
cen = min((yellow(cv2.imread(str(a.out / f"x{x}_y{y}_r0_base.png")), (x, y)) for x, y in done),
          key=lambda v: np.hypot(v[0].mean() - K[0, 2], v[1].mean() - K[1, 2]) if len(v[0]) else 1e9)
size = tuple(float(np.percentile(v, 99) - np.percentile(v, 1)) for v in cen)
print(f"{a.target} seen from below: {size[0]:.0f} x {size[1]:.0f} px")
points, missing = {}, []  # commanded (x, y) -> [tips at rotation 0, tips at rotation 180]
for x, y in done:
    c = [fingers(im, size, (x, y)) if (im := cv2.imread(str(a.out / f"x{x}_y{y}_r{rot}_base.png"))) is not None else None for rot in (0, 180)]
    if c[0] is None:
        missing.append((x, y))
        continue
    points[(x, y)] = c
(a.out / "points.json").write_text(json.dumps([[*k, *v[0], *(v[1] or (None, None))] for k, v in points.items()]))
# The gantry axes lie along the picture's: px depends on y only, py on x only. So each commanded y gets the median px
# over its column and each x the median py over its row, which removes single bad detections.
px_of = {y: float(np.median([c[0][0] for (_, yy), c in points.items() if yy == y])) for y in {k[1] for k in points}}
py_of = {x: float(np.median([c[0][1] for (xx, _), c in points.items() if xx == x])) for x in {k[0] for k in points}}
scatter = np.array([(c[0][0] - px_of[y], c[0][1] - py_of[x]) for (x, y), c in points.items()])
out_yaml = a.out / f"camera-to-robot-cr{n}.yaml"
# the format CoordinateConverter reads: "(px, py)": [x, y]; the pixel is the two tips' midpoint at rotation 0
out_yaml.write_text("".join(f"({round(px_of[y])}, {round(py_of[x])}):\n- {x}\n- {y}\n" for x, y in points))
conv = CoordinateConverter(str(out_yaml))
res = np.array([np.subtract(conv.x_y_to_px_py(x, y), c[0]) for (x, y), c in points.items()])
axis = np.array([np.subtract(c[1], c[0]) / 2 for c in points.values() if c[1]])  # rotation axis minus tips' midpoint, px
print(f"{len(points)} points; tips not found or cut off by the picture's edge at {missing}")
print("px for y:", {y: round(v) for y, v in sorted(px_of.items())}, "\npy for x:", {x: round(v) for x, v in sorted(py_of.items())})
print(f"single detections around those medians, px: std {scatter.std(0).round(1)}, max {np.abs(scatter).max(0).round(1)}")
print(f"fitted mapping vs single detections, px: mean {np.abs(res).mean():.1f}, max {np.abs(res).max():.1f}")
if len(axis):
    print(f"rotation axis minus tips' midpoint at rotation 0, px: median {np.median(axis, 0).round(1)}, std {axis.std(0).round(1)}")
bg = cv2.imread(str(a.out / f"x{done[0][0]}_y{done[0][1]}_r0_base.png"))
for (x, y), ((px, py), _) in points.items():  # ponytail: drawn on one picture, the arm in it is at the first point
    cv2.circle(bg, (round(px), round(py)), 5, (0, 255, 0), 1)
    cv2.drawMarker(bg, tuple(int(v) for v in conv.x_y_to_px_py(x, y)), (0, 0, 255), cv2.MARKER_CROSS, 8, 1)
cv2.imwrite(str(a.out / "check.jpg"), cv2.resize(bg, None, fx=2, fy=2))
if a.install:
    shutil.copy(out_yaml, cams / f"cam-to-robot-points/{out_yaml.name}")
    print("installed", cams / f"cam-to-robot-points/{out_yaml.name}")
