# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["sam2>=1.1.0", "torch", "torchvision", "numpy", "opencv-python-headless", "huggingface_hub", "pyyaml",
#                 "scikit-learn", "requests", "websocket-client"]
# ///
"""Run SceneTracker live on a cell while the arm travels in x, y only, and measure the real loop time.

  export CLOUDGRIPPER_TOKEN=...
  uv run scripts/debug/perception_live.py --start 0.81,0.31 --via "0.88,0.70 0.58,0.70" [--top-every 1]

THIS MOVES THE ROBOT. It sends move_xy alone (z, rotation and gripper are never commanded), one request at a time,
and stops if a request fails or the reported z drifts. --start is the pose the arm was last commanded to: the
reported x, y trails the real arm and is not used. Each loop: move one --step, read both cameras, tracker.step.
The tracker starts on the first frame, before any move, and refreshes on the last frame, back at the start pose.
Output folder (replayable with perception_replay.py): base_NNNN.jpg, top_NNNN.jpg, frames.jsonl (commanded pose,
statuses, timings), live.mp4 (masks and statuses)."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cloudgripper-api"), str(ROOT)]
from client.cloudgripper_client import GripperRobot  # noqa: E402
from cloudgripper_wm.perception import SceneTracker, draw_states  # noqa: E402
from cloudgripper_wm.utils.cloudgripper_image_processor import CloudGripperImageProcessor  # noqa: E402

LO, HI = 0.05, 0.95
p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--robot", default="robot23")
p.add_argument("--start", required=True, help='"x,y" the arm was last commanded to')
p.add_argument("--via", required=True, help='waypoints "x,y x,y ..."; the path ends back at the start')
p.add_argument("--step", type=float, default=0.025, help="x, y travel per loop")
p.add_argument("--period", type=float, default=0.2, help="target seconds per loop (5 Hz)")
p.add_argument("--top-every", type=int, default=1, help="track the top view every N loops (0: only start and end)")
p.add_argument("--out", required=True, help="output folder")
a = p.parse_args()
token = os.environ.get("CLOUDGRIPPER_TOKEN") or sys.exit("set CLOUDGRIPPER_TOKEN")
start = tuple(map(float, a.start.split(",")))
via = [tuple(map(float, w.split(","))) for w in a.via.split()]
assert all(LO <= v <= HI for w in [start, *via] for v in w), f"poses must stay within {LO} to {HI}"
path, cur = [start], np.array(start)
for w in [*via, start]:
    n = max(1, int(np.ceil(np.linalg.norm(np.array(w) - cur) / a.step)))
    path += [tuple(cur + (np.array(w) - cur) * k / n) for k in range(1, n + 1)]
    cur = np.array(w)

cams = ROOT / "cloudgripper_wm/camera_params"
proc = CloudGripperImageProcessor(str(cams / f"cam-to-robot-points/camera-to-robot-cr{a.robot[5:]}.yaml"),
                                  str(cams / f"base-camera-calibration/camera-params-cr{a.robot[5:]}.yaml"))
tracker = SceneTracker(a.robot)
robot = GripperRobot(a.robot, token)
out = Path(a.out); out.mkdir(parents=True)
tmp = tempfile.mkdtemp()
z0, rows = None, []


def describe(states):
    return {n: s.status + (f" on {s.on}" if s.on else "") for n, s in states.items()}


print(f"{a.robot}: {len(path)} loops from {start} -> {out}", flush=True)
with open(out / "frames.jsonl", "w") as log:
    for i, (x, y) in enumerate(path):
        t0 = time.perf_counter()
        x, y = round(float(x), 4), round(float(y), 4)
        if i and robot.move_xy(x, y) is None:
            sys.exit(f"loop {i}: move_xy failed, stopping (arm left where it is)")
        t1 = time.perf_counter()
        state, _, base_raw, _, top, _ = robot.get_all_states()
        if state is None:
            sys.exit(f"loop {i}: getAllStates failed, stopping (arm left where it is)")
        z0 = state["z_norm"] if z0 is None else z0
        if abs(state["z_norm"] - z0) > 0.03:
            sys.exit(f"loop {i}: z moved from {z0:.3f} to {state['z_norm']:.3f}, stopping")
        t2 = time.perf_counter()
        base = proc.undistort_fish_eye(base_raw)
        t3 = time.perf_counter()
        if i == 0:
            states, kind = tracker.start(base, top), "start"
        elif i == len(path) - 1:
            states, kind = tracker.refresh(base, top), "refresh"
        else:
            use_top = a.top_every > 0 and i % a.top_every == 0
            states, kind = tracker.step(base, top if use_top else None), "step2" if use_top else "step1"
        t4 = time.perf_counter()
        row = dict(frame=i, cmd_xy=[x, y], kind=kind, status=describe(states), move_ms=(t1 - t0) * 1e3,
                   camera_ms=(t2 - t1) * 1e3, undistort_ms=(t3 - t2) * 1e3, track_ms=(t4 - t3) * 1e3)
        if kind in ("start", "refresh") or (rows and row["status"] != rows[-1]["status"]):
            print(f"loop {i} ({kind}, {row['track_ms']:.0f} ms): {row['status']}", flush=True)
        cv2.imwrite(str(out / f"base_{i:04d}.jpg"), base, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / f"top_{i:04d}.jpg"), top, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(f"{tmp}/{i:05d}.jpg", draw_states(base, top, states, f"live loop {i} ({kind}) cmd {x:.2f},{y:.2f}"))
        row["loop_ms"] = (time.perf_counter() - t0) * 1e3  # everything, saving included
        rows.append(row); log.write(json.dumps(row) + "\n")
        time.sleep(max(0.0, a.period - (time.perf_counter() - t0)))

steps = [r for r in rows if r["kind"].startswith("step")]
for k in ("move_ms", "camera_ms", "undistort_ms", "track_ms", "loop_ms"):
    v = np.array([r[k] for r in steps])
    print(f"{k:13s} median {np.median(v):6.1f}  95th {np.percentile(v, 95):6.1f}  max {v.max():6.1f}")
print(f"loops over {a.period * 1e3:.0f} ms: {sum(r['loop_ms'] > a.period * 1e3 for r in steps)} of {len(steps)}")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "5", "-i", f"{tmp}/%05d.jpg", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "22", str(out / "live.mp4")], check=True)
print("done; video:", out / "live.mp4")
