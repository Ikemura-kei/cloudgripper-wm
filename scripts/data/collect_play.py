# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["sam2>=1.1.0", "torch", "torchvision", "numpy", "opencv-python", "huggingface_hub", "pyyaml",
#                 "scikit-learn", "requests", "websocket-client", "gymnasium"]
# ///
"""Play-data collection on one cell (cloudgripper_wm/play.py). MOVES THE ROBOT, unattended once started.

  export CLOUDGRIPPER_TOKEN=...
  uv run scripts/data/collect_play.py --out DIR --episodes N [--seed S] [--no-window]

DIR is the session folder; running again with the same DIR continues its episode numbers. Before each episode the
scene check (reset once if needed), the self-check at the start, every 15 episodes and after a surprise; a stop
raises the arm, opens the gripper and exits 1 with the reason (session.jsonl). Ctrl-C stops after the move under way;
the episode in progress then has no steps.npz and is skipped by the conversion.
Convert to training data with scripts/data/play_to_lance.py (parent venv: stable-worldmodel)."""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cloudgripper-api"), str(ROOT)]
import numpy as np  # noqa: E402

from cloudgripper_wm.envs.cloudgripper_env import CloudGripperEnv  # noqa: E402
from cloudgripper_wm.envs.constants import PLAY_MAX_DELTA  # noqa: E402
from cloudgripper_wm.envs.filter_view import SessionView  # noqa: E402
from cloudgripper_wm.envs.filtered_wrapper import FilteredCloudGripperWrapper  # noqa: E402
from cloudgripper_wm.envs.robot_pool import RobotPool  # noqa: E402
from cloudgripper_wm.play import Recorder, collect  # noqa: E402
from cloudgripper_wm.primitives import Primitives  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--robot", default="robot20")
p.add_argument("--out", required=True)
p.add_argument("--episodes", type=int, required=True)
p.add_argument("--seed", type=int, default=None, help="default: from the clock (logged)")
p.add_argument("--no-window", action="store_true", help="no live window (the video is still written)")
a = p.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
seed = a.seed if a.seed is not None else int(np.random.SeedSequence().entropy % 2**31)
RobotPool.configure([a.robot])
env = FilteredCloudGripperWrapper(CloudGripperEnv(max_delta=PLAY_MAX_DELTA, dwell_time=0.5, reset_dwell_time=2.0),
                                  log_path=str(out / "blocked.jsonl"))
view = SessionView(env, out / f"video_{seed}", window=not a.no_window)
rec = Recorder(env, out)
reason = "interrupted"
try:
    env.reset()
    prim = Primitives(env, on_step=lambda label, blocked: (view.show(label, blocked), rec.step(label, blocked)))
    print(f"seed {seed}", flush=True)
    reason = collect(prim, rec, out, a.episodes, np.random.default_rng(seed))
finally:
    view.close(save_pose=True)
    env.close()
    print(f"== collection: {reason}", flush=True)
sys.exit(0 if reason == "done" else 1)
