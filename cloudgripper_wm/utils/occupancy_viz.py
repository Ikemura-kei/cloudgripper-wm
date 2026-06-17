"""Shared 3D live-view helpers for occupancy heightmaps + gripper fingers.

Used by scripts/debug/safety_sim.py and SafeCloudGripperWrapper's
live_view option.
"""

from __future__ import annotations

import numpy as np

from cloudgripper_wm.envs.constants import WORKSPACE_BOUNDS
from cloudgripper_wm.utils.occupancy import HeightMapData
from cloudgripper_wm.utils.get_finger_pos import get_finger_pos

# Height of the translucent wall planes drawn around WORKSPACE_BOUNDS,
# matching scripts/debug/safety_sim.py's WALL_HEIGHT.
WALL_HEIGHT = 0.15

# matplotlib (and the cv2/Qt-plugin fixup it requires) is only needed by
# LiveOccupancyView, and is imported lazily there. Importing it eagerly here
# would pop QT_QPA_PLATFORM_PLUGIN_PATH for every importer of this module
# (e.g. SafeCloudGripperWrapper with live_view=False), breaking cv2.imshow
# elsewhere (e.g. CloudGripperEnv's show_display thread).


def pose_to_fingers(pose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pose = [x, y, z, rotation_norm, gripper_width] -> (left, right, center) xyz."""
    
    x, y, z, rot_norm, w = pose
    theta = rot_norm * np.pi  # rotation_norm in [0,1] -> gripper yaw in [0, 180deg] -> rad
    left, right = get_finger_pos(x, y, z, theta, w)
    return np.asarray(left, dtype=float), np.asarray(right, dtype=float), np.asarray([x, y, z], dtype=float)


def voxel_grid(
    hd: HeightMapData,
    height: float,
    z_cells: int = 3,
    grid_cells: int = 25,
    bounds: tuple[float, float] = (-0.1, 1.1),
):
    """Resample hd.hmap onto a fixed world-frame grid spanning `bounds` in
    both X and Y (robot coordinates are always in [0, 1], so the default
    [-0.1, 1.1] covers the full workspace plus a small margin)."""
    lo, hi = bounds
    span = hi - lo
    cs = span / grid_cells

    centers = lo + (np.arange(grid_cells) + 0.5) * cs
    src_x = np.floor((centers - hd.origin[0]) / hd.cell_size).astype(int)
    src_y = np.floor((centers - hd.origin[1]) / hd.cell_size).astype(int)

    rows, cols = hd.hmap.shape
    valid_x = (src_x >= 0) & (src_x < cols)
    valid_y = (src_y >= 0) & (src_y < rows)

    yy, xx = np.meshgrid(np.arange(grid_cells), np.arange(grid_cells), indexing="ij")
    valid = valid_y[yy] & valid_x[xx]
    vh = np.zeros((grid_cells, grid_cells), dtype=np.float32)  # [y, x]
    vh[valid] = hd.hmap[src_y[yy[valid]], src_x[xx[valid]]]

    dz = height / z_cells
    levels = np.ceil(vh / dz).astype(int)
    kk = np.arange(z_cells)[None, None, :]
    filled = kk < levels.T[:, :, None]

    xc = lo + np.arange(grid_cells + 1) * cs
    yc = lo + np.arange(grid_cells + 1) * cs
    zc = np.linspace(0.0, height, z_cells + 1)
    xg, yg, zg = np.meshgrid(xc, yc, zc, indexing="ij")
    return xg, yg, zg, filled, span, span


class LiveOccupancyView:
    """Interactive 3D view of an occupancy heightmap + gripper finger positions."""

    def __init__(self, height: float, z_cells: int = 3, grid_cells: int = 25):
        import os

        # cv2 points Qt at its own bundled (incompatible) plugins; drop that
        # before matplotlib's Qt backend initializes, or figure creation fails.
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        import matplotlib.pyplot as plt
        self._plt = plt

        self.height = height
        self.z_cells = z_cells
        self.grid_cells = grid_cells
        self.fig = plt.figure(figsize=(7, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        plt.ion()
        plt.show(block=False)

    def update(
        self,
        hd: HeightMapData | None,
        current_pose,
        candidate_pose,
        hit_left: bool = False,
        hit_right: bool = False,
        title: str = "",
    ) -> None:
        ax = self.ax
        ax.cla()

        bounds = (-0.1, 1.1)
        xspan = yspan = bounds[1] - bounds[0]
        if hd is not None:
            xg, yg, zg, filled, xspan, yspan = voxel_grid(hd, self.height, self.z_cells, self.grid_cells, bounds)
            ax.voxels(xg, yg, zg, filled,
                      facecolor=(0.0, 0.7, 0.2, 0.25), edgecolor=(0, 0.4, 0.1, 0.15))

        left0, right0, center0 = pose_to_fingers(current_pose)
        left1, right1, center1 = pose_to_fingers(candidate_pose)

        ax.plot(*zip(center0, center1), color="black", lw=2, label="center")
        ax.plot(*zip(left0, left1), color="red" if hit_left else "tab:blue", lw=2, label="left finger")
        ax.plot(*zip(right0, right1), color="red" if hit_right else "tab:orange", lw=2, label="right finger")
        ax.scatter(*center0, color="black", s=30)
        ax.scatter(*left0, color="tab:blue", s=30)
        ax.scatter(*right0, color="tab:orange", s=30)

        ax.set_xlabel("X (robot)"); ax.set_ylabel("Y (robot)"); ax.set_zlabel("Z (robot)")
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[0], bounds[1])
        ax.set_zlim(0, self.height * 2)
        ax.set_box_aspect((max(xspan, 1e-6), max(yspan, 1e-6), max(xspan * 0.3, 1e-9)))
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(title)

        self.fig.canvas.draw_idle()
        self._plt.pause(0.001)

    def close(self) -> None:
        self._plt.close(self.fig)
        self._plt.ioff()
