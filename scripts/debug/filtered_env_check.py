# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["sam2>=1.1.0", "torch", "torchvision", "numpy", "opencv-python-headless", "huggingface_hub", "pyyaml",
#                 "scikit-learn", "requests", "websocket-client", "gymnasium"]
# ///
"""Supervised check of FilteredCloudGripperWrapper: random actions through the env with the filter on. MOVES THE ROBOT.

  export CLOUDGRIPPER_TOKEN=...
  uv run scripts/debug/filtered_env_check.py --out DIR [--robot robot20] [--steps 40] [--seed 0]

The env resets (arm up, then home: glass centre, top height, closed), then takes --steps random delta actions, each
held for 5 steps and biased downwards so that the fingers reach the objects. Watch it: stop with Ctrl-C.
DIR gets check.mp4 (both views with the masks and each step's verdict) and blocked.jsonl."""
import argparse
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
from cloudgripper_wm.envs.filtered_wrapper import FilteredCloudGripperWrapper  # noqa: E402
from cloudgripper_wm.envs.robot_pool import RobotPool  # noqa: E402
from cloudgripper_wm.perception import draw_states  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--robot", default="robot20")
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=40)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(a.seed)
RobotPool.configure([a.robot])
env = FilteredCloudGripperWrapper(CloudGripperEnv(max_delta=0.05, dwell_time=0.5), log_path=str(out / "blocked.jsonl"))
video = None
try:
    env.reset()
    for i in range(a.steps):
        if i % 5 == 0:
            action = rng.uniform(-0.05, 0.05, 5).astype(np.float32)
            action[2] -= 0.02  # mostly downwards: the arm starts at the top
            action[4] = rng.choice([-0.75, 0.0, 0.0, 0.75])  # now and then close or open fully
        before = env.n_blocked
        env.step(action)
        u = env.unwrapped
        base = env._proc.undistort_fish_eye(u._last_base_img)
        for poly in collision_filter.finger_polys(u._target_pos):
            cv2.polylines(base, [np.int32([env._tracker.conv.x_y_to_px_py(x, y) for x, y in poly])], True, (255, 255, 255), 2)
        pose = "x {:.3f} y {:.3f} z {:.2f} rot {:.0f} grip {:.2f}".format(*u._target_pos[:3], u._target_pos[3] * 180, u._target_pos[4])
        frame = draw_states(base, u._last_top_img, env.states, f"step {i}: {'BLOCKED' if env.n_blocked > before else 'sent'}   |   at {pose}")
        if video is None:
            video = cv2.VideoWriter(str(out / "check.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2, frame.shape[1::-1])
        video.write(frame)
finally:
    if video is not None:
        video.release()
    print(f"{env.n_blocked} of {env.n_steps} steps blocked; arm left at {env.unwrapped._target_pos}")
    env.close()
