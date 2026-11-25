# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["sam2>=1.1.0", "torch", "torchvision", "numpy", "opencv-python", "huggingface_hub", "pyyaml",
#                 "scikit-learn", "requests", "websocket-client", "gymnasium"]
# ///
"""Supervised check of FilteredCloudGripperWrapper: random actions through the env with the filter on. MOVES THE ROBOT.

  export CLOUDGRIPPER_TOKEN=...
  uv run scripts/debug/filtered_env_check.py --out DIR [--robot robot20] [--steps 40] [--seed 0]
  uv run scripts/debug/filtered_env_check.py --out DIR --provoke

The env resets (arm up, then home: glass centre, top height, closed), then takes --steps random delta actions, each
held for 5 steps and biased downwards so that the fingers reach the objects. A window shows both views with the
masks, the finger tips and each step's verdict. Watch it: stop with Ctrl-C in the terminal.
DIR gets check.mp4 (what the window showed, 2 frames per second; written when the script ends) and blocked.jsonl.
--provoke replaces the random actions by deliberate dangerous attempts, to see the filter stop them: the closed
gripper goes over each cube, the slot, the pan and the frame at top height and then lowers in steps of 0.05 until
the filter blocks. Each attempt passes if the block names that object and comes while the tips are still above it.
If a step is sent that brings the tips to within 0.5 mm of the object's top, the filter has failed: the script
raises the arm and stops. The final pose is saved for teleop_filter.py, which can then start without --start."""
import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cloudgripper-api"), str(ROOT)]
from cloudgripper_wm import collision_filter  # noqa: E402
from cloudgripper_wm.envs.cloudgripper_env import CloudGripperEnv  # noqa: E402
from cloudgripper_wm.envs.constants import X_RANGE, Y_RANGE  # noqa: E402
from cloudgripper_wm.envs.filtered_wrapper import FilteredCloudGripperWrapper  # noqa: E402
from cloudgripper_wm.envs.robot_pool import RobotPool  # noqa: E402
from cloudgripper_wm.perception import draw_states  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--robot", default="robot20")
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=40)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--provoke", action="store_true", help="deliberate dangerous attempts instead of random actions")
p.add_argument("--mock", action="store_true", help="no robot, no GPU: mock client and a made-up scene (tests this script's own logic)")
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(a.seed)
RobotPool.configure([a.robot])
tracker = None
if a.mock:  # a cube, the slot and the pan at made-up places; one cube outside the env's x, y range
    from types import SimpleNamespace

    def obj(x, y, w, h):
        return SimpleNamespace(status="on_glass", on=None, xy=(x, y), height_mm=h, mask_base=None, mask_top=None, inferred_px=None,
                               footprint_xy=np.array([x, y]) + w / 2 * collision_filter.MM * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]))
    scene = dict(cube_blue=obj(0.7, 0.34, 10, 10), cube_white=obj(0.1, 0.3, 10, 10), slot=obj(0.3, 0.55, 20, 16), pan=obj(0.5, 0.7, 25, 7))
    tracker = SimpleNamespace(start=lambda b, t: scene, step=lambda b, t: scene, refresh=lambda b, t: scene)
env = FilteredCloudGripperWrapper(CloudGripperEnv(max_delta=0.05, dwell_time=0 if a.mock else 0.5, reset_dwell_time=0 if a.mock else 2.0, use_mock=a.mock),
                                  log_path=str(out / "blocked.jsonl"), tracker=tracker)
video = None


def step(action, label):
    """One env step, shown and recorded; True if the filter blocked it."""
    global video
    before = env.n_blocked
    env.step(np.asarray(action, np.float32))
    u, blocked = env.unwrapped, env.n_blocked > before
    base = env._proc.undistort_fish_eye(u._last_base_img)
    for poly in collision_filter.finger_polys(u._target_pos) if not a.mock else ():
        cv2.polylines(base, [np.int32([env._tracker.conv.x_y_to_px_py(x, y) for x, y in poly])], True, (255, 255, 255), 2)
    pose = "x {:.3f} y {:.3f} z {:.2f} rot {:.0f} grip {:.2f}".format(*u._target_pos[:3], u._target_pos[3] * 180, u._target_pos[4])
    frame = draw_states(base, u._last_top_img, env.states, f"{label}: {'BLOCKED' if blocked else 'sent'}   |   at {pose}")
    if video is None:
        video = cv2.VideoWriter(str(out / "check_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2, frame.shape[1::-1])
    video.write(frame)
    if not a.mock:
        cv2.imshow(f"filtered env check, {a.robot}", cv2.resize(frame, None, fx=0.75, fy=0.75))
        cv2.waitKey(1)
    return blocked


def provoke():
    """Lower the closed gripper onto each object and the frame; the filter has to stop every attempt."""
    u = env.unwrapped
    off = np.add(collision_filter.TOOL_OFFSET, collision_filter.TOOL_OFFSET_TURNING)  # tips' midpoint - commanded x, y at rotation 0
    targets = [(n, np.array(s.xy), s.height_mm) for n, s in env.states.items() if n != "rope" and s.status == "on_glass"]
    targets.append(("the frame", np.array([1.0, 0.5]), collision_filter.FRAME_MM))  # x is clipped to the env's range: tips over the frame
    results = []
    for name, xy, top in targets:
        goal = np.array([*(xy - off), 1.0, 0.0, u._gripper_range[0]])
        if name == "the frame":
            goal[0] = X_RANGE[1]
        elif not (X_RANGE[0] <= goal[0] <= X_RANGE[1] and Y_RANGE[0] <= goal[1] <= Y_RANGE[1]):
            results.append((name, "skipped: outside the env's x, y range", float("nan")))
            continue
        if step(goal - u._target_pos, f"to above {name}"):  # at top height, closed
            results.append((name, "skipped: the way there was blocked", float("nan")))
            continue
        outcome = "reached the lowest height unblocked"
        while u._target_pos[2] > 0:
            n_before = len(open(out / "blocked.jsonl").readlines()) if (out / "blocked.jsonl").exists() else 0
            if step([0, 0, -0.05, 0, 0], f"lowering onto {name}"):
                reason = json.loads(open(out / "blocked.jsonl").readlines()[n_before])["reason"]
                outcome = "PASS" if name in reason else f"blocked for another reason ({reason})"
                break
            if collision_filter.tip_mm(u._target_pos[2]) < top + 0.5:
                step([0, 0, 1.0, 0, 0], "FILTER FAILED, raising")
                sys.exit(f"the filter let the tips down to {collision_filter.tip_mm(u._target_pos[2]):.1f} mm over {name} ({top:g} mm tall): stopped")
        results.append((name, outcome, collision_filter.tip_mm(u._target_pos[2])))
        step([0, 0, 1.0 - u._target_pos[2], 0, 0], f"up from {name}")
    for name, outcome, tip in results:
        print(f"  {name}: {outcome}" + ("" if np.isnan(tip) else f" (tips stopped at {tip:.1f} mm)"))


try:
    env.reset()
    if a.provoke:
        provoke()
    for i in range(0 if a.provoke else a.steps):
        if i % 5 == 0:
            action = rng.uniform(-0.05, 0.05, 5).astype(np.float32)
            action[2] -= 0.02  # mostly downwards: the arm starts at the top
            action[4] = rng.choice([-0.75, 0.0, 0.0, 0.75])  # now and then close or open fully
        step(action, f"step {i}")
finally:
    if video is not None:  # re-encode to H.264 so that VS Code, browsers and chat apps play it
        video.release()
        if subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(out / "check_raw.mp4"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out / "check.mp4")]).returncode == 0:
            (out / "check_raw.mp4").unlink()
    last = Path.home() / f".cache/cloudgripper_wm/{a.robot}_last_pose.json"
    last.parent.mkdir(parents=True, exist_ok=True)
    (Path("/dev/null") if a.mock else last).write_text(json.dumps([round(float(v), 4) for v in env.unwrapped._target_pos]))
    print(f"{env.n_blocked} of {env.n_steps} steps blocked; arm left at {env.unwrapped._target_pos}")
    env.close()
