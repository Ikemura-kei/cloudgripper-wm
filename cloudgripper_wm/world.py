"""Thin wrapper around swm.World that handles CloudGripper-specific setup.

Usage (1 robot):
    world = CloudGripperWorld(robot_names=["robot23"])
    world.set_policy(RandomPolicy())
    world.collect("data.lance", episodes=10)

Usage (8 robots — identical call site):
    world = CloudGripperWorld(robot_names=["robot1", ..., "robot8"])
    world.collect("data.lance", episodes=100)

Usage (with collision-avoidance safety filter):
    world = CloudGripperWorld(robot_names=["robot23"], safety={"cell_size": 0.01, "height": 0.35})

`safety`, if set, wraps each env in `SafeCloudGripperWrapper` (with
`live_view=False`) so collected `"action"` values are the actually-executed
(post-safety-filter) actions, not the policy's raw output.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any

import stable_worldmodel as swm

import cloudgripper_wm.envs  # triggers gymnasium.register() for all env IDs
from cloudgripper_wm.envs.robot_pool import RobotPool
from cloudgripper_wm.envs.safe_cloudgripper_wrapper import SafeCloudGripperWrapper


class CloudGripperWorld:
    """swm.World pre-configured for CloudGripper.

    Handles RobotPool setup and token resolution so the caller only needs to
    specify which robots to use and collection parameters.
    """

    def __init__(
        self,
        robot_names: list[str],
        env_name: str = "cloudgripper/Gripper-v0",
        image_shape: tuple[int, int] = (64, 64),
        max_episode_steps: int = 100,
        token: str | None = None,
        use_mock: bool = False,
        dwell_time: float = 0.5,
        max_delta: float = 0.05,
        safety: dict | None = None,
        **kwargs: Any,
    ) -> None:
        if not use_mock:
            token = token or os.environ.get("CLOUDGRIPPER_TOKEN")
            if not token:
                raise RuntimeError("Set CLOUDGRIPPER_TOKEN or pass token=")

        RobotPool.configure(robot_names)

        extra_wrappers = kwargs.pop("extra_wrappers", [])
        if safety is not None:
            extra_wrappers = [
                *extra_wrappers,
                partial(SafeCloudGripperWrapper, live_view=False, **safety),
            ]

        self._world = swm.World(
            env_name,
            num_envs=len(robot_names),
            image_shape=image_shape,
            max_episode_steps=max_episode_steps,
            token=token,
            use_mock=use_mock,
            dwell_time=dwell_time,
            max_delta=max_delta,
            extra_wrappers=extra_wrappers,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Delegate swm.World interface
    # ------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self._world.num_envs

    @property
    def infos(self) -> dict:
        return self._world.infos

    def set_policy(self, policy: Any) -> None:
        self._world.set_policy(policy)

    def reset(self, **kwargs: Any) -> None:
        self._world.reset(**kwargs)

    def collect(self, *args: Any, **kwargs: Any) -> Any:
        return self._world.collect(*args, **kwargs)

    def evaluate(self, **kwargs: Any) -> dict:
        return self._world.evaluate(**kwargs)

    def close(self) -> None:
        self._world.close()
