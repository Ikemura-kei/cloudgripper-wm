# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["sam2>=1.1.0", "torch", "torchvision", "numpy", "opencv-python", "huggingface_hub", "pyyaml",
#                 "scikit-learn", "requests", "websocket-client", "gymnasium"]
# ///
"""Supervised run of play primitives (cloudgripper_wm/primitives.py) through the filtered env. MOVES THE ROBOT.

  export CLOUDGRIPPER_TOKEN=...
  uv run scripts/debug/primitives_check.py --out DIR PRIMITIVE [PRIMITIVE ...]

A PRIMITIVE is a name with its arguments after colons; they run in the order given, after one env reset (arm up,
then home):
  reset                          bring objects that are far from the glass centre back towards it
  push:pan:1,0:30                push the pan 30 mm along robot (x, y) = (1, 0)
  pick_place:cube_white:0.4,0.5  grasp the white cube and put it down at x 0.4, y 0.5
  cube_into_slot:cube_blue       grasp the blue cube and drop it into the slot's notch
  rope_drag:0:0.1,0              grasp the rope near end 0 (or 1) and drag it by (0.1, 0)

Every move is sent as delta actions within PLAY_MAX_DELTA (x, y 0.1, height 0.25, rotation 0.25, gripper 0.75 per
step). A window shows both views with the masks, the tips and each move's verdict. Watch it: Ctrl-C in the terminal stops
after the move in progress (the arm stays where it is). DIR gets check.mp4 and blocked.jsonl; the final pose is saved
for teleop_filter.py. --mock runs the same without robot or GPU on a made-up scene, to test a command line."""
import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cloudgripper-api"), str(ROOT)]
from cloudgripper_wm.envs.cloudgripper_env import CloudGripperEnv  # noqa: E402
from cloudgripper_wm.envs.constants import PLAY_MAX_DELTA  # noqa: E402
from cloudgripper_wm.envs.filter_view import SessionView, mock_scene  # noqa: E402
from cloudgripper_wm.envs.filtered_wrapper import FilteredCloudGripperWrapper  # noqa: E402
from cloudgripper_wm.envs.robot_pool import RobotPool  # noqa: E402
from cloudgripper_wm.primitives import Primitives  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("primitives", nargs="+")
p.add_argument("--robot", default="robot20")
p.add_argument("--out", required=True)
p.add_argument("--mock", action="store_true", help="no robot, no GPU: mock client and a made-up scene")
a = p.parse_args()


def parse(spec):
    """'push:pan:1,0:30' -> ('push', ['pan', (1.0, 0.0), 30.0])"""
    name, *args = spec.split(":")
    conv = lambda v: tuple(map(float, v.split(","))) if "," in v else float(v) if v.replace(".", "", 1).lstrip("-").isdigit() else v
    args = [conv(v) for v in args]
    if name == "rope_drag":
        args[0] = int(args[0])
    return {"reset": "reset_objects"}.get(name, name), args


assert parse("push:pan:1,0:30") == ("push", ["pan", (1.0, 0.0), 30.0]) and parse("rope_drag:0:0.1,0") == ("rope_drag", [0, (0.1, 0.0)])
todo = [parse(s) for s in a.primitives]
unknown = [n for n, _ in todo if n not in ("reset_objects", "push", "pick_place", "cube_into_slot", "rope_drag")]
if unknown:
    sys.exit(f"unknown primitive(s): {unknown}")

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
tracker = None
if a.mock:
    from types import SimpleNamespace
    scene = mock_scene()
    tracker = SimpleNamespace(start=lambda b, t: scene, step=lambda b, t: scene, refresh=lambda b, t: scene)
RobotPool.configure([a.robot])
env = FilteredCloudGripperWrapper(CloudGripperEnv(max_delta=PLAY_MAX_DELTA, dwell_time=0 if a.mock else 0.5, reset_dwell_time=0 if a.mock else 2.0, use_mock=a.mock),
                                  log_path=str(out / "blocked.jsonl"), tracker=tracker)
view = SessionView(env, out, window=not a.mock)
t0 = time.time()
try:
    env.reset()
    prim = Primitives(env, on_step=view.show)
    for (name, args), spec in zip(todo, a.primitives):
        result = getattr(prim, name)(*args)
        print(f"== {spec}: {result if isinstance(result, dict) else 'done' if result else 'GIVEN UP'}", flush=True)
finally:
    view.close(save_pose=not a.mock)
    print(f"{env.n_blocked} of {env.n_steps} steps blocked, {time.time() - t0:.0f} s in all; arm left at {env.unwrapped._target_pos}")
    env.close()
