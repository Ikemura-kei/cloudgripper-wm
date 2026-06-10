"""Safety wrapper for CloudGripperEnv.

Before forwarding an action to the underlying env, predicts the gripper's
finger-tip positions for the current and post-action target pose and checks
both against an occupancy heightmap built from the previous step's base
camera image. If either finger's path would cross the top surface of an
object, the action is replaced with a zero action (target position held in
place) and a warning is printed. Otherwise the action passes through
unchanged and the wrapper is a no-op.

Usage
-----
    import gymnasium as gym
    from cloudgripper_wm.envs.safe_cloudgripper_wrapper import SafeCloudGripperWrapper

    env = gym.make("cloudgripper/Gripper-v0", ...)
    env = SafeCloudGripperWrapper(env, live_view=True)
"""

from __future__ import annotations

import os
import re

import cv2
import numpy as np
import gymnasium as gym

from cloudgripper_wm.envs.constants import GRIPPER_RANGE, Y_RANGE
from cloudgripper_wm.utils.cloudgripper_image_processor import CloudGripperImageProcessor
from cloudgripper_wm.utils.occupancy import HeightMapData, build_height_map, check_collision
from cloudgripper_wm.utils.occupancy_viz import LiveOccupancyView, pose_to_fingers


_CAM_PARAMS_DIR = os.path.join(os.path.dirname(__file__), "..", "camera_params")


def _calibration_paths(robot_name: str) -> tuple[str, str]:
    m = re.match(r"robot(\d+)", robot_name)
    if not m:
        raise ValueError(f"Cannot derive camera calibration id from robot name {robot_name!r}")
    cr_id = f"cr{m.group(1)}"
    cam_to_robot = os.path.join(_CAM_PARAMS_DIR, "cam-to-robot-points", f"camera-to-robot-{cr_id}.yaml")
    cam_params = os.path.join(_CAM_PARAMS_DIR, "base-camera-calibration", f"camera-params-{cr_id}.yaml")
    if not (os.path.exists(cam_to_robot) and os.path.exists(cam_params)):
        raise FileNotFoundError(
            f"No camera calibration found for {robot_name!r} "
            f"(expected {cam_to_robot} and {cam_params})"
        )
    return cam_to_robot, cam_params


class SafeCloudGripperWrapper(gym.Wrapper):
    """Gymnasium wrapper that blocks actions predicted to cause a collision.

    On every step:
      1. Build an occupancy heightmap from the cached base image (from the
         previous step / reset).
      2. Predict left/right finger-tip positions for the current and
         post-action target pose.
      3. If either finger's straight-line path crosses the top surface of an
         occupied cell, replace the action with a zero action (target pose
         unchanged) and print a warning. Otherwise pass the action through.
    """

    def __init__(
        self,
        env: gym.Env,
        cell_size: float = 0.01,
        height: float = 0.35,
        live_view: bool = False,
        **height_map_kwargs,
    ) -> None:
        super().__init__(env)
        self._robot_name: str = self.unwrapped._robot_name
        cam_to_robot_yaml, cam_params_yaml = _calibration_paths(self._robot_name)
        self._image_processor = CloudGripperImageProcessor(cam_to_robot_yaml, cam_params_yaml)

        self._cell_size = cell_size
        self._height = height
        self._height_map_kwargs = height_map_kwargs

        self._last_hmap: HeightMapData | None = None
        self._view: LiveOccupancyView | None = LiveOccupancyView(height) if live_view else None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[dict, dict]:
        obs, info = self.env.reset(**kwargs)
        self._update_heightmap(info)
        if self._view is not None:
            pos = self.unwrapped._target_pos
            self._view.update(self._last_hmap, pos, pos, title=f"{self._robot_name} (reset)")
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        current_pos = self.unwrapped._target_pos.copy()
        candidate_pos = np.clip(current_pos + action, 0.0, 1.0)
        candidate_pos[1] = np.clip(candidate_pos[1], *Y_RANGE)
        candidate_pos[4] = np.clip(candidate_pos[4], *GRIPPER_RANGE)

        hit_left = hit_right = False
        if self._last_hmap is not None:
            hit_left, hit_right = self._check_collision(current_pos, candidate_pos)
            if hit_left or hit_right:
                print("!" * 70)
                print("!! SAFETY: predicted collision with object top surface")
                print(f"!!   left finger collision={hit_left}, right finger collision={hit_right}")
                print("!!   forcing zero action (holding current target position)")
                print("!" * 70)
                action = np.zeros_like(action)
                candidate_pos = current_pos

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_heightmap(info)

        if self._view is not None:
            new_pos = self.unwrapped._target_pos
            title = f"{self._robot_name}"
            if hit_left or hit_right:
                title += "  [BLOCKED - collision predicted]"
            self._view.update(self._last_hmap, current_pos, new_pos, hit_left, hit_right, title=title)

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._view is not None:
            self._view.close()
        self.env.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_heightmap(self, info: dict) -> None:
        base_img_rgb = info.get("pixels_base")
        if base_img_rgb is None:
            return
        base_img_bgr = cv2.cvtColor(base_img_rgb, cv2.COLOR_RGB2BGR)
        try:
            self._last_hmap = build_height_map(
                self._image_processor, base_img_bgr,
                cell_size=self._cell_size, height=self._height,
                **self._height_map_kwargs,
            )
        except ValueError:
            self._last_hmap = None  # no objects detected — nothing to check against

    def _check_collision(self, current_pos: np.ndarray, candidate_pos: np.ndarray) -> tuple[bool, bool]:
        left0, right0, _ = pose_to_fingers(current_pos)
        left1, right1, _ = pose_to_fingers(candidate_pos)
        # Only the top face of each object is treated as an obstacle —
        # lateral contact (pushing an object from the side) is allowed.
        hit_left, _ = check_collision(left0, left1, self._last_hmap, surface="top")
        hit_right, _ = check_collision(right0, right1, self._last_hmap, surface="top")
        return hit_left, hit_right
