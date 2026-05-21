"""Interactive test script for CloudGripperEnv on real hardware.

Usage:
    uv run python scripts/test_real_robot.py
    uv run python scripts/test_real_robot.py --robot robot2 --steps 20 --dwell 0.8

Requires CLOUDGRIPPER_TOKEN env var to be set.
Press any key to advance to the next step, 'q' to quit early.
"""

import argparse
import os

import cv2
import numpy as np

from cloudgripper_wm.envs.cloudgripper_env import CloudGripperEnv
from cloudgripper_wm.envs.robot_pool import RobotPool


def show(title: str, img: np.ndarray) -> bool:
    """Display a BGR image. Returns False if user pressed 'q'."""
    cv2.imshow(title, img)
    key = cv2.waitKey(30) & 0xFF
    return key != ord("q")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="robot23", help="Robot name (robot1–robot32)")
    parser.add_argument("--token", default=None, help="API token (falls back to CLOUDGRIPPER_TOKEN)")
    parser.add_argument("--steps", type=int, default=10, help="Number of random steps")
    parser.add_argument("--dwell", type=float, default=0.1, help="Seconds to wait after each command")
    parser.add_argument("--max-delta", type=float, default=0.05, help="Max action delta per step")
    args = parser.parse_args()

    token = args.token or os.environ.get("CLOUDGRIPPER_TOKEN")
    if not token:
        raise SystemExit("Set CLOUDGRIPPER_TOKEN or pass --token")

    RobotPool.configure([args.robot])

    print(f"Connecting to {args.robot} ...")
    env = CloudGripperEnv(
        token=token,
        max_delta=args.max_delta,
        dwell_time=args.dwell,
        render_mode="rgb_array",
    )

    print("Resetting to home position ...")
    obs, info = env.reset()
    print(f"  state: {obs['state']}")

    top = env.render()
    base = info["pixels_base"]
    print(f"  top camera:  {top.shape} {top.dtype}")
    print(f"  base camera: {base.shape} {base.dtype}")

    rng = np.random.default_rng()

    for step in range(args.steps):
        action = env.action_space.sample()
        # action[:] = 0
        obs, reward, terminated, truncated, info = env.step(action)
        top = env.render()
        base = info["pixels_base"]

        print(f"step {step + 1:>3}/{args.steps}  state={np.round(obs['state'], 3)}  action={np.round(action, 3)}")

        title_top = f"top"
        if not show(title_top, top):
            break
        show(f"base", base)

        if terminated or truncated:
            print("Episode ended early.")
            break

    env.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
