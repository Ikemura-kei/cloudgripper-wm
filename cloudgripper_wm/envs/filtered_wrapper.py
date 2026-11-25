"""Collision filter inside the env: every action is checked against the tracked objects before it is sent.

    env = FilteredCloudGripperWrapper(gym.make("cloudgripper/Gripper-v0", ...))
    # or: CloudGripperWorld(robot_names=["robot20"], collision_filter={})

On reset the arm is raised first (the env's own reset travels to the home pose at the height the arm is at), then
`SceneTracker.start` finds the objects. On every step `collision_filter.check` judges the commanded pose against
the objects as seen after the previous step; a blocked action is replaced by a zero action, as in
SafeCloudGripperWrapper, so the recorded action is the executed one. Needs a CUDA GPU and the `perception` extra.

ponytail: `collision_filter.use_cell` is module-wide, so all robots of one process must be the same cell type
(today: robot20 only, or robot23 only). Give the filter a per-cell object when two cells run together.
"""
from __future__ import annotations

import json
import time

import gymnasium as gym
import numpy as np

from cloudgripper_wm import collision_filter
from cloudgripper_wm.envs.constants import GRIPPER_RANGE
from cloudgripper_wm.envs.safe_cloudgripper_wrapper import _calibration_paths
from cloudgripper_wm.utils.cloudgripper_image_processor import CloudGripperImageProcessor

# Gripper command range per cell (closed, open). Cell 20: Kei, 2026-11-25.
CELL_GRIPPER_RANGE = {"robot20": (0.1, 0.825)}
# Commanded x, y range per cell. Cell 20 (Kei, 2026-11-25): beyond the frame's inner edges (glass 0.043 to 0.842 /
# 0.052 to 0.845), so that objects at the frame can be pushed back; the filter guards the frame. 0.05 to 0.95 is
# the arm's range as used so far. On the low sides that is no further than the glass edge: nothing can get behind
# an object lying against the frame there. Beyond x 0.88 nothing is visible from below.
CELL_XY_RANGE = {"robot20": ((0.05, 0.95), (0.05, 0.95))}


class FilteredCloudGripperWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, log_path: str | None = None, tracker=None) -> None:
        """`log_path`: a .jsonl that gets one line per blocked action. `tracker`: for tests (default: SceneTracker)."""
        super().__init__(env)
        u = self.unwrapped
        self._name: str = u._robot_name
        collision_filter.use_cell(self._name)
        u._gripper_range = CELL_GRIPPER_RANGE.get(self._name, GRIPPER_RANGE)
        u._x_range, u._y_range = CELL_XY_RANGE.get(self._name, (u._x_range, u._y_range))
        self._proc = CloudGripperImageProcessor(*_calibration_paths(self._name))
        if tracker is None:
            from cloudgripper_wm.perception import SceneTracker  # here: it loads SAM 2 onto the GPU
            tracker = SceneTracker(self._name)
        self._tracker, self._log_path = tracker, log_path
        self.states: dict | None = None
        self.held: list[str] = []  # objects in the grasp, from the filter's verdicts
        self.n_steps = self.n_blocked = 0

    def _see(self, start: bool = False) -> None:
        u = self.unwrapped
        base, top = self._proc.undistort_fish_eye(u._last_base_img), u._last_top_img
        self.states = self._tracker.start(base, top) if start else self._tracker.step(base, top)
        if any(s.status == "unaccounted" for s in self.states.values()):  # blocks every move: look again once
            self.states = self._tracker.refresh(base, top)

    def reset(self, **kwargs):
        u = self.unwrapped
        if u._target_pos[2] < 1.0 or not np.isfinite(u._last_sent_pos[2]):  # up first, then the env's travel home
            (u.robot.move_z_ws if u.use_ws else u.robot.move_z)(1.0)
            time.sleep(u.reset_dwell_time)
        obs, info = self.env.reset(**kwargs)
        self._see(start=True)
        lost = [n for n, s in self.states.items() if s.status == "unaccounted"]
        if lost:
            print(f"[{self._name}] filter: {', '.join(lost)} not found at reset; every move is blocked until they are")
        return obs, info

    def step(self, action):
        u = self.unwrapped
        action = np.asarray(action, dtype=np.float32)
        current = u._target_pos.copy()
        candidate = np.clip(current + action, 0.0, 1.0)  # as CloudGripperEnv.step clips it
        if u._restrict_xy:
            candidate[0], candidate[1] = np.clip(candidate[0], *u._x_range), np.clip(candidate[1], *u._y_range)
        candidate[4] = np.clip(candidate[4], *u._gripper_range)
        verdict = collision_filter.check(current, candidate, self.states, self.held)
        self.n_steps += 1
        if not verdict.allowed:
            self.n_blocked += 1
            print(f"[{self._name}] filter: {verdict.reason}; holding the pose ({self.n_blocked} of {self.n_steps} steps blocked)")
            if self._log_path:
                with open(self._log_path, "a") as f:
                    f.write(json.dumps(dict(t=time.time(), pose0=current.tolist(), pose1=candidate.tolist(), reason=verdict.reason)) + "\n")
            action = np.zeros_like(action)
        else:
            self.held = verdict.held
        out = self.env.step(action)
        self._see()
        return out
