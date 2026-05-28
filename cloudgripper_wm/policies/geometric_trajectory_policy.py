"""Geometric trajectory policy — traces circle, square, or triangle in X-Y space.

The X-Y trajectory follows the configured shape; Z, rotation, and gripper use
sticky-random motion (same mechanism as StickyRandomPolicy).

All absolute positions are pre-computed and clipped to [0, 1] before conversion
to delta actions so the robot stays within the workspace bounds.
"""

import numpy as np
from stable_worldmodel.policy import BasePolicy

_VALID_SHAPES = ("circle", "square", "triangle")


def _build_xy_waypoints(
    shape: str,
    size: float,
    center: tuple[float, float],
    steps_per_loop: int,
) -> np.ndarray:
    """Return absolute (x, y) waypoints for one full loop, clipped to [0, 1].

    Returns shape (steps_per_loop, 2).
    """
    cx, cy = center
    r = size / 2.0

    if shape == "circle":
        angles = np.linspace(0, 2 * np.pi, steps_per_loop, endpoint=False)
        xs = cx + r * np.cos(angles)
        ys = cy + r * np.sin(angles)

    elif shape == "square":
        half = r
        corners = np.array([
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ])
        pts = []
        steps_per_side = steps_per_loop // 4
        for i in range(4):
            a = corners[i]
            b = corners[(i + 1) % 4]
            t = np.linspace(0, 1, steps_per_side, endpoint=False)
            seg = a + t[:, None] * (b - a)
            pts.append(seg)
        pts = np.concatenate(pts, axis=0)
        # pad/trim to exact steps_per_loop
        if len(pts) < steps_per_loop:
            pts = np.concatenate([pts, pts[:steps_per_loop - len(pts)]], axis=0)
        pts = pts[:steps_per_loop]
        xs, ys = pts[:, 0], pts[:, 1]

    elif shape == "triangle":
        angles = np.array([np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3])
        corners = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1)
        pts = []
        steps_per_side = steps_per_loop // 3
        for i in range(3):
            a = corners[i]
            b = corners[(i + 1) % 3]
            t = np.linspace(0, 1, steps_per_side, endpoint=False)
            seg = a + t[:, None] * (b - a)
            pts.append(seg)
        pts = np.concatenate(pts, axis=0)
        if len(pts) < steps_per_loop:
            pts = np.concatenate([pts, pts[:steps_per_loop - len(pts)]], axis=0)
        pts = pts[:steps_per_loop]
        xs, ys = pts[:, 0], pts[:, 1]

    else:
        raise ValueError(f"shape must be one of {_VALID_SHAPES}, got {shape!r}")

    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)
    return np.stack([xs, ys], axis=1).astype(np.float32)


class GeometricTrajectoryPolicy(BasePolicy):
    """Policy that moves the robot's end-effector along a geometric shape in X-Y.

    X-Y motion follows the configured shape; Z, rotation, and gripper use the
    same sticky-random mechanism as StickyRandomPolicy.

    Args:
        shape: One of "circle", "square", "triangle".
        size: Side length / diameter of the shape in normalized [0, 1] coords.
        center: (x, y) center of the shape in normalized coords.
        steps_per_loop: Number of env steps for one full traversal of the shape.
        min_repeat: Minimum sticky-random repeat for Z/rot/grip DoFs.
        max_repeat: Maximum sticky-random repeat for Z/rot/grip DoFs.
        noise_std: Gaussian noise std added to sticky-random DoFs each step.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        shape: str = "circle",
        size: float = 0.3,
        center: tuple[float, float] = (0.5, 0.5),
        steps_per_loop: int = 60,
        min_repeat: int = 3,
        max_repeat: int = 8,
        noise_std: float = 0.05,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if shape not in _VALID_SHAPES:
            raise ValueError(f"shape must be one of {_VALID_SHAPES}, got {shape!r}")
        self.shape = shape
        self.size = size
        self.center = center
        self.steps_per_loop = steps_per_loop
        self.min_repeat = min_repeat
        self.max_repeat = max_repeat
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self._waypoints: np.ndarray = _build_xy_waypoints(shape, size, center, steps_per_loop)
        self._step_idx: int = 0
        self._current_pos_xy: np.ndarray = np.array([0.5, 0.5], dtype=np.float32)

        self._sticky_base: np.ndarray | None = None  # values for z, rot, grip DoFs
        self._sticky_remaining: int = 0

    def get_action(self, obs, **kwargs) -> np.ndarray:
        # Flatten to (5,) — EnvPool wraps action space as (num_envs, 5)
        lo = np.asarray(self.env.action_space.low).reshape(-1)[:5]
        hi = np.asarray(self.env.action_space.high).reshape(-1)[:5]

        # --- X-Y: move toward current target waypoint ---
        # Advance to the next waypoint only once the current one is reachable in
        # a single step — this ensures the robot fully transits from home to the
        # first waypoint before the trajectory index starts incrementing.
        target_wp = self._waypoints[self._step_idx % self.steps_per_loop]
        delta_to_target = target_wp - self._current_pos_xy
        if np.all(np.abs(delta_to_target) <= hi[:2]):
            self._step_idx += 1
        dxy = np.clip(delta_to_target, lo[:2], hi[:2])
        self._current_pos_xy = np.clip(self._current_pos_xy + dxy, 0.0, 1.0)

        # --- sticky-random for z, rotation, gripper (indices 2, 3, 4) ---
        if self._sticky_remaining <= 0:
            self._sticky_base = self.rng.uniform(lo[2:], hi[2:]).astype(np.float32)
            self._sticky_remaining = int(
                self.rng.integers(self.min_repeat, self.max_repeat + 1)
            )
        noise = self.rng.normal(0.0, self.noise_std, size=3).astype(np.float32)
        d_other = np.clip(self._sticky_base + noise, lo[2:], hi[2:])
        self._sticky_remaining -= 1

        action = np.concatenate([dxy, d_other]).astype(np.float32)
        return action.reshape(self.env.action_space.low.shape)

    def reset(self) -> None:
        self._step_idx = 0
        self._current_pos_xy = np.array([0.5, 0.5], dtype=np.float32)
        self._sticky_base = None
        self._sticky_remaining = 0
