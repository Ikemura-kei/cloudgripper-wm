import os
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from cloudgripper_wm.envs.robot_pool import RobotPool
from cloudgripper_wm.tasks import get_task
from cloudgripper_wm.tasks.base import DEFAULT_HOME_POS, Task


class CloudGripperEnv(gym.Env):
    """Single-robot CloudGripper Gymnasium environment.

    Acquires a robot name from RobotPool on init and releases it on close().
    Configure the pool before creating any envs:

        RobotPool.configure(["robot23"])          # 1 robot
        RobotPool.configure(["robot1", ..., "robot8"])  # 8 robots

    Images are exposed via render() (top camera → "pixels" via MegaWrapper).
    The observation dict only carries "state" so MegaWrapper's
    EverythingToInfoWrapper never collides with the "pixels" key AddPixelsWrapper
    already wrote into info.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(
        self,
        token: str | None = None,
        task: str | None = None,
        max_delta: float = 0.05,
        dwell_time: float = 0.5,
        render_mode: str | None = None,
        use_mock: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_delta = max_delta
        self.dwell_time = dwell_time
        self.task: Task | None = get_task(task)

        self._robot_name = RobotPool.acquire()

        if use_mock:
            from client.cloudgripper_client_mock import GripperRobotMock
            self.robot = GripperRobotMock(self._robot_name, token or "mock")
        else:
            from client.cloudgripper_client import GripperRobot
            tok = token or os.environ["CLOUDGRIPPER_TOKEN"]
            self.robot = GripperRobot(self._robot_name, tok)

        self.action_space = spaces.Box(
            low=-max_delta,
            high=max_delta,
            shape=(5,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32),
        })

        self._target_pos: np.ndarray = DEFAULT_HOME_POS.copy()
        self._last_top_img: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Core Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        self._target_pos = (
            self.task.home_pos().copy() if self.task is not None else DEFAULT_HOME_POS.copy()
        )
        self._send_absolute()
        # gripper open on reset regardless of rotation/grip target
        self.robot.gripper_open()

        time.sleep(self.dwell_time)

        top_img, _ = self.robot.getImageTop()
        base_img, _ = self.robot.getImageBase()
        self._last_top_img = top_img

        obs = {"state": self._target_pos.copy()}
        info = {"pixels_base": base_img}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        self._target_pos = np.clip(self._target_pos + action, 0.0, 1.0)
        self._send_absolute()

        time.sleep(self.dwell_time)

        top_img, _ = self.robot.getImageTop()
        base_img, _ = self.robot.getImageBase()
        self._last_top_img = top_img

        obs = {"state": self._target_pos.copy()}
        info: dict = {"pixels_base": base_img}

        if self.task is not None:
            reward = self.task.compute_reward(obs, action, info)
            terminated = self.task.check_terminated(obs, info)
            info["success"] = self.task.check_success(obs, info)
            info.update(self.task.get_task_info(obs))
        else:
            reward = 0.0
            terminated = False

        return obs, reward, terminated, False, info

    def render(self) -> np.ndarray:
        if self._last_top_img is None:
            top_img, _ = self.robot.getImageTop()
            self._last_top_img = top_img
        return cv2.cvtColor(self._last_top_img, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        RobotPool.release(self._robot_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_absolute(self) -> None:
        x, y, z, rot_norm, grip = self._target_pos
        self.robot.move_xy(float(x), float(y))
        self.robot.move_z(float(z))
        self.robot.rotate(int(float(rot_norm) * 180))
        self.robot.move_gripper(float(grip))
