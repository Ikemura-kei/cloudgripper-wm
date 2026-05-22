"""Data collection entry point.

Usage:
    uv run python scripts/collect.py
    uv run python scripts/collect.py --robots robot23 --episodes 5 --output data/test.lance
    uv run python scripts/collect.py --mock --episodes 2 --output data/mock.lance
"""

import argparse
import os

from stable_worldmodel.policy import RandomPolicy

from cloudgripper_wm.world import CloudGripperWorld


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robots", nargs="+", default=["robot23"],
        help="Robot names to use (e.g. --robots robot1 robot2 robot3)",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--dwell", type=float, default=0.5)
    parser.add_argument("--max-delta", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output", default="data/collect.lance")
    parser.add_argument("--mock", action="store_true", help="Use mock robots (no hardware)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Robots:   {args.robots}")
    print(f"Episodes: {args.episodes}  Steps: {args.steps}  Mock: {args.mock}")
    print(f"Output:   {args.output}")

    world = CloudGripperWorld(
        robot_names=args.robots,
        image_shape=(args.image_size, args.image_size),
        max_episode_steps=args.steps,
        use_mock=args.mock,
        dwell_time=args.dwell,
        max_delta=args.max_delta,
    )

    world.set_policy(RandomPolicy(seed=args.seed))
    world.collect(args.output, episodes=args.episodes, seed=args.seed)
    world.close()
    print("Done.")


if __name__ == "__main__":
    main()
