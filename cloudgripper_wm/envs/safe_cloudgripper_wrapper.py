"""Safety wrapper for CloudGripperEnv.

Before forwarding an action to the underlying env, predicts the gripper's
finger-tip positions for the current and post-action target pose and checks
both against an occupancy heightmap built from the previous step's base
camera image. If either finger's path would cross the top surface of an
object, the action is replaced with a zero action (target position held in
place) and a warning is printed. Otherwise the action passes through
unchanged and the wrapper is a no-op.

NOTE: object detection assumes GREEN objects (see the default
`hsv_lower`/`hsv_upper` in `utils.occupancy.build_height_map`). Objects of
other colors will not be picked up by the heightmap and will not be avoided.

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
import time
from typing import Callable

import cv2
import numpy as np
import gymnasium as gym

from cloudgripper_wm.envs.constants import GRIPPER_RANGE, WORKSPACE_BOUNDS, X_RANGE, Y_RANGE
from cloudgripper_wm.utils.cloudgripper_image_processor import CloudGripperImageProcessor
from cloudgripper_wm.utils.occupancy import (
    HeightMapData,
    build_height_map,
    check_collision,
    is_inside_occupancy,
    object_near_wall,
)
from cloudgripper_wm.utils.occupancy_viz import LiveOccupancyView, pose_to_fingers


# Maps the edge returned by `object_near_wall` to the (axis, sign) of
# "moving toward that edge" — e.g. moving toward "x_min" means x decreases.
_WALL_EDGE_DIRECTION = {
    "x_min": (0, -1.0),
    "x_max": (0, 1.0),
    "y_min": (1, -1.0),
    "y_max": (1, 1.0),
}

# Each side of the workspace, for push_objects_to_center: the axis held at
# the wall edge, the edge's value, the sign to push inward along that axis,
# and the axis swept along the edge.
_PUSH_SIDES = [
    (0, 0.0, 1.0, 1),   # x_min wall: x=0, push toward +x, sweep over y
    (0, 1.0, -1.0, 1),  # x_max wall: x=1, push toward -x, sweep over y
    (1, 0.0, 1.0, 0),   # y_min wall: y=0, push toward +y, sweep over x
    (1, 1.0, -1.0, 0),  # y_max wall: y=1, push toward -y, sweep over x
]


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
        grid_cells: int = 25,
        wall_margin: float = 0.21,
        reset_objects_every: int = 10,
        cooldown_every: int = 12,
        cooldown_time: float = 20.0,
        **height_map_kwargs,
    ) -> None:
        super().__init__(env)
        self._robot_name: str = self.unwrapped._robot_name
        cam_to_robot_yaml, cam_params_yaml = _calibration_paths(self._robot_name)
        self._image_processor = CloudGripperImageProcessor(cam_to_robot_yaml, cam_params_yaml)

        self._cell_size = cell_size
        self._height = height
        self._wall_margin = wall_margin
        self._height_map_kwargs = height_map_kwargs
        self._reset_objects_every = reset_objects_every
        self._cooldown_every = cooldown_every
        self._cooldown_time = cooldown_time
        self._episode_count = 0

        self._last_hmap: HeightMapData | None = None
        self._view: LiveOccupancyView | None = (
            LiveOccupancyView(height, grid_cells=grid_cells) if live_view else None
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[dict, dict]:
        obs, info = self.env.reset(**kwargs)
        self._update_heightmap(info)

        self._episode_count += 1
        # print(f"episode {self._episode_count} reset")
        if self._episode_count == 1 or (self._reset_objects_every > 0 and self._episode_count % self._reset_objects_every == 0):
            print(f"[{self._robot_name}] episode {self._episode_count}: running push_objects_to_center()")
            self.push_objects_to_center()
            # Return to home position after rearranging the workspace.
            obs, info = self.env.reset(**kwargs)
            self._update_heightmap(info)

        if self._cooldown_every > 0 and self._episode_count % self._cooldown_every == 0:
            print(f"[{self._robot_name}] episode {self._episode_count}: cooling down for {self._cooldown_time}s")
            time.sleep(self._cooldown_time)

        if self._view is not None:
            pos = self.unwrapped._target_pos
            self._view.update(self._last_hmap, pos, pos, title=f"{self._robot_name} (reset)")
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        current_pos = self.unwrapped._target_pos.copy()
        candidate_pos = np.clip(current_pos + action, 0.0, 1.0)
        if self.unwrapped._restrict_xy:
            candidate_pos[0] = np.clip(candidate_pos[0], *X_RANGE)
            candidate_pos[1] = np.clip(candidate_pos[1], *Y_RANGE)
        candidate_pos[4] = np.clip(candidate_pos[4], *GRIPPER_RANGE)

        hit_left = hit_right = False
        if self._last_hmap is not None:
            hit_left, hit_right = self._check_collision(current_pos, candidate_pos)
            if hit_left or hit_right:
                print("!" * 70)
                print("!! SAFETY: predicted collision (object top surface or wall-jam)")
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

    def push_objects_to_center(
        self,
        push_amount: float = 0.25,
        n_sweeps: int = 7,
        transit_z: float = 0.5,
        push_z: float = 0.2,
        dwell_time: float = 0.5,
        on_step: Callable[[dict, dict], None] | None = None,
    ) -> None:
        """Sweep the gripper around the four workspace edges, pushing any
        objects near the walls toward the center.

        For each side of the workspace, visits `n_sweeps` evenly spaced
        positions along that edge. At each position: move to the edge at
        `transit_z`, descend to `push_z`, push `push_amount` inward (gripper
        closed, rotation 0), then retreat back to `transit_z` before moving
        to the next position. Each of these is a single absolute move
        (`_move_to`), with collision/wall-jam checks run on the full move's
        finger trajectory.

        If `on_step` is given, it's called as `on_step(obs, info)` after
        every `step()` — useful for updating a live camera view.

        The X_RANGE/Y_RANGE clamps (which restrict normal operation to a
        smaller-than-full workspace) are temporarily disabled for the
        duration of this routine, since the edges/pushes need to reach the
        full WORKSPACE_BOUNDS to retrieve objects near the walls.

        `dwell_time` temporarily overrides `CloudGripperEnv.dwell_time` for
        the duration of this routine — these moves are larger than typical
        per-step deltas, so the robot needs more settling time before the
        next observation.
        """
        sweep_vals = np.linspace(0.0, 1.0, n_sweeps)
        prev_restrict_xy = self.unwrapped._restrict_xy
        prev_dwell_time = self.unwrapped.dwell_time
        self.unwrapped._restrict_xy = False
        self.unwrapped.dwell_time = dwell_time
        try:
            for edge_axis, edge_val, push_sign, sweep_axis in _PUSH_SIDES:
                push_target = edge_val + push_sign * push_amount
                for sweep_val in sweep_vals:
                    approach = np.array([0.0, 0.0, transit_z, 0.0, 0.0], dtype=np.float32)
                    approach[edge_axis] = edge_val
                    approach[sweep_axis] = sweep_val
                    self._move_to(approach, on_step=on_step)

                    descend = approach.copy()
                    descend[2] = push_z
                    self._move_to(descend, on_step=on_step)

                    push = descend.copy()
                    push[edge_axis] = push_target
                    self._move_to(push, on_step=on_step)

                    retreat = push.copy()
                    retreat[2] = transit_z
                    self._move_to(retreat, on_step=on_step)
        finally:
            self.unwrapped._restrict_xy = prev_restrict_xy
            self.unwrapped.dwell_time = prev_dwell_time

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _move_to(
        self,
        target_pos: np.ndarray,
        tol: float = 1e-3,
        on_step: Callable[[dict, dict], None] | None = None,
    ) -> None:
        """Move directly to `target_pos` in a single `step()` call.

        The underlying env sends absolute commands per axis
        (`_send_absolute`), so one large delta is one robot move + one
        dwell — no need to chunk by `max_delta` as in normal RL rollouts.
        Collision/wall-jam checks operate on the full current->target
        segment, so this is equivalent (not weaker) than many small steps.
        """
        current = self.unwrapped._target_pos
        delta = target_pos - current
        if np.allclose(delta, 0.0, atol=tol):
            return
        obs, _, _, _, info = self.step(delta.astype(np.float32))
        if on_step is not None:
            on_step(obs, info)
        if np.allclose(self.unwrapped._target_pos, current, atol=1e-6):
            print(f"[{self._robot_name}] _move_to: action blocked, target {target_pos} not reached")

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

        # `check_collision` only fires on the initial top-crossing. If a
        # finger has already entered an occupied column (e.g. via lateral
        # movement, or because `height` overestimates the real object), it
        # won't catch further descent — block that here.
        if not hit_left and left1[2] < left0[2] and is_inside_occupancy(left0, self._last_hmap):
            hit_left = True
        if not hit_right and right1[2] < right0[2] and is_inside_occupancy(right0, self._last_hmap):
            hit_right = True

        # Block sustained pushing of an object that's already jammed against
        # the workspace wall — a small gap to the wall is fine, but pressing
        # further into it (without lateral retreat) risks gripper damage.
        if not hit_left and self._pushes_into_wall(left0, left1):
            hit_left = True
        if not hit_right and self._pushes_into_wall(right0, right1):
            hit_right = True

        return hit_left, hit_right

    def _pushes_into_wall(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        # Only a finger that's actually at the object's level (in contact,
        # not flying over it) can be "pushing" it — same condition as
        # is_inside_occupancy's z <= top check.
        if not is_inside_occupancy(p1, self._last_hmap):
            return False
        edge = object_near_wall(p1[:2], self._last_hmap, WORKSPACE_BOUNDS, self._wall_margin)
        if edge is None:
            return False
        axis, sign = _WALL_EDGE_DIRECTION[edge]
        # Block only if the finger is moving toward the wall (or staying put)
        # along that axis — retreating away from the wall is always allowed.
        return sign * (p1[axis] - p0[axis]) >= 0.0
