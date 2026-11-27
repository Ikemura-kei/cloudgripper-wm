# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["sam2>=1.1.0", "torch", "torchvision", "numpy", "opencv-python", "huggingface_hub", "pyyaml",
#                 "scikit-learn", "requests", "websocket-client", "gymnasium"]
# ///
"""Self-check of a play cell (cloudgripper_wm/self_check.py): every axis, the gripper's look and tilt, the cameras.
MOVES THE ROBOT, through the collision filter, never below height 0.35.

  export CLOUDGRIPPER_TOKEN=...
  uv run scripts/self_check.py --out DIR --record   # once, on the healthy robot: the reference, then commit it
  uv run scripts/self_check.py --out DIR            # a check against it

The reference is cloudgripper_wm/self_check_ref/<robot> unless --ref names another directory.

Exit status: 0 pass (or recorded), 1 fail, 2 a move was blocked by the filter (the scene is in the way, not a robot
fault). The window shows every move as primitives_check.py does. DIR keeps everything for diagnosis: check.mp4,
blocked.jsonl, scene_start.npz (the tracker's states with masks), raw/ (every picture, retakes included), log.jsonl
(one line per picture with the numbers), one overlay per pose (reference-only masks magenta, this check's cyan, both
white; "_FAIL" where the pose failed) and report.txt."""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "third_party/cloudgripper-api"), str(ROOT)]
from cloudgripper_wm.envs.cloudgripper_env import CloudGripperEnv  # noqa: E402
from cloudgripper_wm.envs.constants import PLAY_MAX_DELTA  # noqa: E402
from cloudgripper_wm.envs.filter_view import SessionView  # noqa: E402
from cloudgripper_wm.envs.filtered_wrapper import FilteredCloudGripperWrapper  # noqa: E402
from cloudgripper_wm.envs.robot_pool import RobotPool  # noqa: E402
from cloudgripper_wm.primitives import Primitives  # noqa: E402
from cloudgripper_wm.self_check import self_check  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--robot", default="robot20")
p.add_argument("--out", required=True)
p.add_argument("--ref", help="reference directory (default: cloudgripper_wm/self_check_ref/<robot>)")
p.add_argument("--record", action="store_true", help="record the reference (healthy robot only)")
a = p.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
RobotPool.configure([a.robot])
env = FilteredCloudGripperWrapper(CloudGripperEnv(max_delta=PLAY_MAX_DELTA, dwell_time=0.5, reset_dwell_time=2.0),
                                  log_path=str(out / "blocked.jsonl"))
view = SessionView(env, out)
status = "blocked"
try:
    env.reset()
    status, lines = self_check(Primitives(env, on_step=view.show), a.ref, out=out, record=a.record)
    print("\n".join(lines))
    print(f"== self-check: {status.upper()}", flush=True)
finally:
    view.close(save_pose=True)
    env.close()
sys.exit({"pass": 0, "recorded": 0, "fail": 1}.get(status, 2))
