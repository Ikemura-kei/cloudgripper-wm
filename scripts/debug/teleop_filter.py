# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["sam2>=1.1.0", "torch", "torchvision", "numpy", "opencv-python", "huggingface_hub", "pyyaml",
#                 "scikit-learn", "requests", "websocket-client"]
# ///
"""Teleoperate cell 20 (default) or 23 one move at a time, to test the collision filter by eye. THIS MOVES THE ROBOT.

  export CLOUDGRIPPER_TOKEN=...
  uv run scripts/debug/teleop_filter.py [--robot robot20] [--start X,Y,Z,ROT_DEG[,GRIP]] --out DIR [--finger-sep-mm 23] [--step 0.025] [--jam-mm 5] [--contact-mm 4]

A key proposes a move; nothing is sent yet. The view then shows the finger tips where the move would put them
(green: touches nothing, yellow: pushes a free object, red: blocked) with the filter's verdict in the bottom line:

  a / d   x - / +        r / f   up / down        y   send the proposed move
  w / s   y + / -        q / e   rotation + / -   Y   send it although the filter blocks it
  p       look again for objects not on the glass (tracker.refresh; park the arm out of the way first)
  o / c   open / close the gripper (cell 20): proposed and checked like any other move
  x, Esc  quit           any other key takes the proposed move back

--start is the pose the arm was last COMMANDED to (x, y, height, rotation in degrees): the robot's reported pose
trails the arm, and its reported height does not follow at all (2026-11-24). Without it the script starts from the
pose it last commanded itself (saved in ~/.cache/cloudgripper_wm/); pass it again after anything else moved the arm.
GRIP is the gripper command the fingers are at (cell 20: 0.1 closed, the default, to 0.825 open); the filter takes
the tip spacing from it. Cell 23's gripper is never commanded: --finger-sep-mm tells the filter its spacing (23 open,
7 closed). Only the axis that changes is sent
(move_xy, move_z or rotate), one request at a time. x, y stay within 0.05 to 0.95.

DIR gets moves.jsonl (every proposal: poses, verdict, sent or taken back), the view at each decision, and a video of
the whole session as shown in the window (teleop_<time>.mp4, 4 frames per second).
--no-window N runs N camera + tracking loops without a window or any move, and exits: a connection check."""
import argparse
import atexit
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cloudgripper-api"), str(ROOT)]
from client.cloudgripper_client import GripperRobot  # noqa: E402
from cloudgripper_wm import collision_filter  # noqa: E402
from cloudgripper_wm.perception import SceneTracker, draw_states  # noqa: E402
from cloudgripper_wm.utils.cloudgripper_image_processor import CloudGripperImageProcessor  # noqa: E402

LIMITS = [(0.05, 0.95), (0.05, 0.95), (0.0, 1.0), (0.0, 1.0)]  # x, y, height, rotation / 180
GRIP = {"o": 0.825, "c": 0.1}  # cell 20
KEYS = {"a": (0, -1), "d": (0, 1), "w": (1, 1), "s": (1, -1), "r": (2, 1), "f": (2, -1), "q": (3, 1), "e": (3, -1)}


def propose(pose, key, steps):
    """pose [x, y, z, rotation / 180, gripper] after one press of `key`, kept within LIMITS."""
    new = list(pose)
    if key in GRIP:
        new[4] = GRIP[key]
        return new
    axis, sign = KEYS[key]
    new[axis] = round(float(np.clip(pose[axis] + sign * steps[axis], *LIMITS[axis])), 4)
    return new


assert propose([0.5, 0.5, 0.5, 0.0, 0.1], "d", [0.025, 0.025, 0.05, 0.05]) == [0.525, 0.5, 0.5, 0.0, 0.1]
assert propose([0.5, 0.5, 0.5, 0.0, 0.1], "o", [0.025] * 4)[4] == 0.825
assert propose([0.94, 0.5, 0.02, 0.0], "d", [0.025] * 4)[0] == 0.95 and propose([0.94, 0.5, 0.02, 0.0], "f", [0.025, 0.025, 0.05, 0.05])[2] == 0.0

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--robot", default="robot20")
p.add_argument("--start", default=None, help='"x,y,z,rotation_deg" the arm was last commanded to (default: where this script left it)')
p.add_argument("--out", required=True, help="output folder")
p.add_argument("--step", type=float, default=0.025, help="x, y step per key (height 0.05, rotation 9 deg)")
p.add_argument("--finger-sep-mm", type=float, default=None, help="finger tip spacing for the filter (open 23, closed 7)")
p.add_argument("--jam-mm", type=float, default=None, help=f"how close to the frame a pushed object may end up (default {collision_filter.JAM_MM:g})")
p.add_argument("--contact-mm", type=float, default=None, help=f"sideways: a finger this close to an object counts as touching (default {collision_filter.CONTACT_MM:g})")
p.add_argument("--scale", type=float, default=0.75, help="window size relative to the 2240x890 view")
p.add_argument("--no-window", type=int, default=0, metavar="N", help="N loops without window or moves, then exit")
a = p.parse_args()
token = os.environ.get("CLOUDGRIPPER_TOKEN") or sys.exit("set CLOUDGRIPPER_TOKEN")
# The pose this script last commanded, kept across runs. It is wrong if anything else moved the arm since.
last = Path.home() / f".cache/cloudgripper_wm/{a.robot}_last_pose.json"
if a.start:
    x, y, z, rot, *g = map(float, a.start.split(","))
    pose = [x, y, z, rot / 180, *g]
elif last.exists():
    pose = json.loads(last.read_text())
    print(f"starting from the pose this script left the arm at ({time.ctime(last.stat().st_mtime)}): x {pose[0]} y {pose[1]} "
          f"z {pose[2]} rot {pose[3] * 180:.0f}. If anything else moved the arm since, quit and pass --start.", flush=True)
else:
    sys.exit("no saved pose yet: pass --start X,Y,Z,ROT_DEG once")
pose = pose[:4] + (pose[4:] or [GRIP["c"]])  # [x, y, z, rotation / 180, gripper]
last.parent.mkdir(parents=True, exist_ok=True)
last.write_text(json.dumps(pose))
assert all(lo <= v <= hi for v, (lo, hi) in zip(pose, LIMITS)), f"--start outside {LIMITS}"
steps = [a.step, a.step, 0.05, 0.05]
collision_filter.use_cell(a.robot)
if a.finger_sep_mm:
    collision_filter.FINGER_SEP_MM = a.finger_sep_mm
if a.jam_mm is not None:
    collision_filter.JAM_MM = a.jam_mm
if a.contact_mm is not None:
    collision_filter.CONTACT_MM = a.contact_mm

cams = ROOT / "cloudgripper_wm/camera_params"
proc = CloudGripperImageProcessor(str(cams / f"cam-to-robot-points/camera-to-robot-cr{a.robot[5:]}.yaml"),
                                  str(cams / f"base-camera-calibration/camera-params-cr{a.robot[5:]}.yaml"))
tracker = SceneTracker(a.robot)
robot = GripperRobot(a.robot, token)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
win = f"teleop with collision filter, {a.robot}"


def look(refresh=False):
    state, _, base_raw, _, top, _ = robot.get_all_states()
    if state is None:
        sys.exit("getAllStates failed, stopping (arm left where it is)")
    base = proc.undistort_fish_eye(base_raw)
    if not tracker.__dict__.get("states"):
        return base, top, tracker.start(base, top)
    return base, top, tracker.refresh(base, top) if refresh else tracker.step(base, top)


def full(q):  # filter pose: [x, y, z, rotation / 180, gripper]
    return q


def send(old, new):
    """Send only the axes that changed, in the env's order; False if a request failed."""
    ok = True
    if new[:2] != old[:2]:
        ok &= robot.move_xy(new[0], new[1]) is not None
    if new[2] != old[2]:
        ok &= robot.move_z(new[2]) is not None
    if new[3] != old[3]:
        ok &= robot.rotate(int(round(new[3] * 180))) is not None
    if new[4] != old[4]:
        ok &= robot.move_gripper(new[4]) is not None
    return ok


def view(base, top, states, pending, verdict, note):
    base = base.copy()
    for q, colour, w in ((pose, (255, 255, 255), 2),) + (((pending, (0, 0, 255) if not verdict.allowed else (0, 255, 255) if verdict.pushed else (0, 255, 0), 2),) if pending else ()):
        for poly in collision_filter.finger_polys(full(q)):
            cv2.polylines(base, [np.int32([tracker.conv.x_y_to_px_py(px, py) for px, py in poly])], True, colour, w)
    here = f"at x {pose[0]:.3f} y {pose[1]:.3f} z {pose[2]:.2f} ({collision_filter.tip_mm(pose[2]):.0f} mm) rot {pose[3] * 180:.0f} grip {pose[4]}"
    if pending:
        what = "BLOCKED" if not verdict.allowed else "PUSH (allowed)" if verdict.pushed else "FREE"
        label = f"{what}: {verdict.reason}   |   y = send{' (Y to force)' if not verdict.allowed else ''}, other key = take back   |   {here}"
    else:
        label = f"{note}   |   {here}   |   white boxes: finger tips now"
    return draw_states(base, top, states, label)


print("""
keys (click the window first):
  a / d   x - / +        r / f   up / down        y   send the proposed move
  w / s   y + / -        q / e   rotation + / -   Y   send it although the filter blocks it
  p       look again for objects not on the glass
  o / c   open / close the gripper
  x, Esc  quit           any other key takes the proposed move back
""", flush=True)
note, pending, verdict, n = "press a move key", None, None, 0
raw, video = out / "teleop_raw.mp4", None


@atexit.register  # also after a failed request
def finish_video():
    if video is not None and video.isOpened():  # re-encode to H.264 so that browsers and chat apps play it
        video.release()
        mp4 = out / f"teleop_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        if subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)]).returncode == 0:
            raw.unlink()
        print(f"video: {mp4 if mp4.exists() else raw}", flush=True)


with open(out / "moves.jsonl", "a") as log:
    while True:
        base, top, states = look()
        frame = view(base, top, states, pending, verdict, note)
        if a.no_window:
            n += 1
            print(f"loop {n}: " + "; ".join(f"{k} {s.status}" + (f" on {s.on}" if s.on else "") for k, s in states.items()), flush=True)
            if n >= a.no_window:
                cv2.imwrite(str(out / "no_window.jpg"), frame)
                break
            continue
        if video is None:
            video = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 4, frame.shape[1::-1])
        video.write(frame)
        cv2.imshow(win, cv2.resize(frame, None, fx=a.scale, fy=a.scale))
        k = cv2.waitKey(30) & 0xFF
        key = chr(k) if k < 128 else ""
        if key in ("x", "\x1b") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if pending and key:
            go = key == "y" and verdict.allowed or key == "Y"
            row = dict(t=time.time(), pose0=pose, pose1=pending, tip_mm=collision_filter.tip_mm(pending[2]), allowed=verdict.allowed,
                       reason=verdict.reason, pushed=verdict.pushed, decision="sent" if go else "taken back", forced=key == "Y" and not verdict.allowed,
                       status={k_: s.status + (f" on {s.on}" if s.on else "") for k_, s in states.items()})
            cv2.imwrite(str(out / f"move_{int(row['t'] * 10)}.jpg"), frame)
            if go and not send(pose, pending):
                row["decision"] = "request failed"
                log.write(json.dumps(row) + "\n")
                sys.exit("a move request failed, stopping (arm left where it is)")
            log.write(json.dumps(row) + "\n"); log.flush()
            note = f"{row['decision']}: {verdict.reason}"
            pose, pending, verdict = (pending if go else pose), None, None
            last.write_text(json.dumps(pose))
        elif key in KEYS or key in GRIP and a.robot == "robot20":
            pending = propose(pose, key, steps)
            if pending == pose:
                pending, note = None, "at the limit"
            else:
                verdict = collision_filter.check(full(pose), full(pending), states)
        elif key == "p":
            look(refresh=True)
            note = "looked again"
cv2.destroyAllWindows()
