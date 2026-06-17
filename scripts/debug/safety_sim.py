"""Animate gripper finger-tip trajectories against a synthetic occupancy
heightmap, covering every case SafeCloudGripperWrapper's collision check
handles:

  - top-surface collision (descending onto an object)
  - lateral push within an object's footprint (no top-crossing -> clear)
  - "already inside" edge case (continued descent inside an occupied column,
    without a fresh top-crossing)
  - pass through free space (clear)
  - wall-jam guard (pushing an object that's already against the workspace
    wall -> blocked; retreating -> clear; pushing an object away from any
    wall -> clear)
  - flying over an object near the wall at a safe height (no contact ->
    clear, even though the wall-jam guard's (x,y) check alone would match)

Heightmap layout (20x20 grid, cell_size=0.05, covering WORKSPACE_BOUNDS):
  - Object A: x in [0, 0.15], y in [0.4, 0.6]   -- flush against the x_min wall
  - Object B: x in [0.45, 0.6], y in [0.4, 0.6] -- away from any wall

The workspace wall (just outside WORKSPACE_BOUNDS) is drawn as translucent
blue planes; occupied heightmap cells are drawn as translucent green voxels.
A finger whose move is blocked is drawn in red and held at its start
position for the whole animation (mirrors the wrapper replacing the action
with a zero action).

Usage
-----
    uv run python scripts/debug/safety_sim.py
"""

import os

import numpy as np

from cloudgripper_wm.envs.constants import WORKSPACE_BOUNDS
from cloudgripper_wm.utils.occupancy import (
    HeightMapData,
    check_collision,
    is_inside_occupancy,
    object_near_wall,
)
from cloudgripper_wm.utils.occupancy_viz import pose_to_fingers, voxel_grid

# occupancy.py imports cv2, which points Qt at its own bundled (incompatible)
# plugins; drop that *after* the cv2 import above but *before* matplotlib's
# Qt backend initializes, or figure creation fails.
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


HEIGHT = 0.35    # assumed object height, in robot z units [0, 1]
CELL = 0.05      # heightmap grid cell size, in robot x/y units [0, 1]
WALL_MARGIN = 0.05
WALL_HEIGHT = 0.15

N_FRAMES = 30
FRAME_PAUSE = 0.05
HOLD_PAUSE = 1.0
HOLD_FRAMES = round(HOLD_PAUSE / FRAME_PAUSE)

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "misc", "safety_sim.mp4")
VIDEO_FPS = round(1.0 / FRAME_PAUSE)

# Mirrors SafeCloudGripperWrapper._WALL_EDGE_DIRECTION: maps the edge
# returned by object_near_wall to the (axis, sign) of "moving toward that
# edge" — e.g. moving toward "x_min" means x decreases.
_WALL_EDGE_DIRECTION = {
    "x_min": (0, -1.0),
    "x_max": (0, 1.0),
    "y_min": (1, -1.0),
    "y_max": (1, 1.0),
}


def build_synthetic_heightmap() -> HeightMapData:
    """20x20 grid covering WORKSPACE_BOUNDS exactly (cell_size=0.05).

    - Object A: cols 0-2 (x in [0, 0.15]), rows 8-11 (y in [0.4, 0.6])
      -> flush against the x_min wall.
    - Object B: cols 9-11 (x in [0.45, 0.6]), rows 8-11 (y in [0.4, 0.6])
      -> away from every wall.
    """
    lo, hi = WORKSPACE_BOUNDS
    n = int(round((hi - lo) / CELL))
    hmap = np.zeros((n, n), dtype=np.float32)
    hmap[8:12, 0:3] = HEIGHT   # object A — against x_min wall
    hmap[8:12, 9:12] = HEIGHT  # object B — away from walls
    return HeightMapData(hmap=hmap, origin=(lo, lo), cell_size=CELL)


def evaluate_scenario(hd: HeightMapData, scenario: dict) -> dict:
    """Replicates SafeCloudGripperWrapper._check_collision for both fingers."""
    start = np.asarray(scenario["start"], dtype=float)
    end = np.asarray(scenario["end"], dtype=float)

    left0, right0, _ = pose_to_fingers(start)
    left1, right1, _ = pose_to_fingers(end)

    result = {}
    for name, p0, p1 in (("left", left0, left1), ("right", right0, right1)):
        hit, hit_pt = check_collision(p0, p1, hd, surface="top")
        reason = "top_crossing" if hit else None

        if not hit and p1[2] < p0[2] and is_inside_occupancy(p0, hd):
            hit = True
            reason = "already_inside_descent"

        if not hit and is_inside_occupancy(p1, hd):
            edge = object_near_wall(p1[:2], hd, WORKSPACE_BOUNDS, WALL_MARGIN)
            if edge is not None:
                axis, sign = _WALL_EDGE_DIRECTION[edge]
                if sign * (p1[axis] - p0[axis]) >= 0.0:
                    hit = True
                    reason = f"wall_jam({edge})"

        result[name] = {"blocked": hit, "reason": reason, "hit_point": hit_pt}

    print(f"\n{scenario['label']}")
    for name in ("left", "right"):
        r = result[name]
        print(f"  {name:>5} finger: blocked={r['blocked']}  reason={r['reason']}")

    return result


def draw_walls(ax, bounds_plot: tuple[float, float]) -> None:
    """Draw the workspace wall as translucent blue planes just outside
    WORKSPACE_BOUNDS, around the full plotted extent."""
    lo, hi = WORKSPACE_BOUNDS
    plo, phi = bounds_plot
    zz = np.array([0.0, WALL_HEIGHT])

    # x = lo and x = hi walls, spanning y over the plotted extent
    for x in (lo, hi):
        yy, z_grid = np.meshgrid([plo, phi], zz)
        x_grid = np.full_like(yy, x)
        ax.plot_surface(x_grid, yy, z_grid, color="tab:blue", alpha=0.2, shade=False)

    # y = lo and y = hi walls, spanning x over the plotted extent
    for y in (lo, hi):
        xx, z_grid = np.meshgrid([plo, phi], zz)
        y_grid = np.full_like(xx, y)
        ax.plot_surface(xx, y_grid, z_grid, color="tab:blue", alpha=0.2, shade=False)


def build_scenarios() -> list[dict]:
    # Pose = [x, y, z, rotation_norm, gripper]. rotation_norm=0, gripper=1.0
    # -> fingers are offset along X by ~0.114 (left = center - 0.114,
    # right = center + 0.114).
    return [
        {
            "label": "high pass over object B (expect clear - above object height)",
            "start": [0.30, 0.50, 0.60, 0.0, 1.0],
            "end":   [0.75, 0.50, 0.60, 0.0, 1.0],
        },
        {
            "label": "lateral push within object B (expect clear - side push allowed)",
            "start": [0.60, 0.50, 0.10, 0.0, 1.0],
            "end":   [0.65, 0.50, 0.10, 0.0, 1.0],
        },
        {
            "label": "descend onto object B (expect collision - top surface)",
            "start": [0.60, 0.50, 0.60, 0.0, 1.0],
            "end":   [0.60, 0.50, 0.05, 0.0, 1.0],
        },
        {
            "label": "pass through free space (expect clear)",
            "start": [0.05, 0.20, 0.10, 0.0, 1.0],
            "end":   [0.25, 0.20, 0.10, 0.0, 1.0],
        },
        {
            "label": "already inside object B, continue descending (expect collision - inside edge case)",
            "start": [0.60, 0.50, 0.20, 0.0, 1.0],
            "end":   [0.60, 0.50, 0.10, 0.0, 1.0],
        },
        {
            "label": "in contact: push object A into x_min wall (expect blocked - wall jam)",
            "start": [0.25, 0.50, 0.20, 0.0, 1.0],
            "end":   [0.20, 0.50, 0.20, 0.0, 1.0],
        },
        {
            "label": "in contact: retreat from x_min wall (expect clear - retreat always allowed)",
            "start": [0.19, 0.50, 0.20, 0.0, 1.0],
            "end":   [0.21, 0.50, 0.20, 0.0, 1.0],
        },
        {
            "label": "in contact: push object B, away from any wall (expect clear)",
            "start": [0.70, 0.50, 0.20, 0.0, 1.0],
            "end":   [0.65, 0.50, 0.20, 0.0, 1.0],
        },
        {
            "label": "flying over object A near x_min wall (expect clear - no contact)",
            "start": [0.25, 0.50, 0.60, 0.0, 1.0],
            "end":   [0.20, 0.50, 0.60, 0.0, 1.0],
        },
    ]


def animate_scenario(fig, ax, hd: HeightMapData, scenario: dict, result: dict, writer: FFMpegWriter | None) -> None:
    start = np.asarray(scenario["start"], dtype=float)
    end = np.asarray(scenario["end"], dtype=float)

    poses = start[None, :] * (1 - np.linspace(0, 1, N_FRAMES))[:, None] \
        + end[None, :] * np.linspace(0, 1, N_FRAMES)[:, None]
    lefts, rights, centers = zip(*(pose_to_fingers(p) for p in poses))
    lefts, rights, centers = np.array(lefts), np.array(rights), np.array(centers)

    bounds_plot = (-0.1, 1.1)
    xg, yg, zg, filled, xspan, yspan = voxel_grid(hd, HEIGHT, z_cells=3, grid_cells=25, bounds=bounds_plot)

    ax.cla()
    ax.voxels(xg, yg, zg, filled,
              facecolor=(0.0, 0.7, 0.2, 0.25), edgecolor=(0, 0.4, 0.1, 0.15))
    draw_walls(ax, bounds_plot)
    ax.set_xlabel("X (robot)"); ax.set_ylabel("Y (robot)"); ax.set_zlabel("Z (robot)")
    ax.set_zlim(0, HEIGHT * 2)
    ax.set_box_aspect((xspan, yspan, max(xspan * 0.3, 1e-9)))
    ax.set_title(scenario["label"])

    def hit_frame(p0, p1, r):
        if not r["blocked"]:
            return None
        if r["reason"] != "top_crossing" or r["hit_point"] is None:
            # Held from the start — the wrapper replaces the whole action
            # with zero, so the finger never leaves p0.
            return 0
        seg = p1 - p0
        denom = float(np.dot(seg, seg))
        t = 0.0 if denom == 0 else float(np.dot(r["hit_point"] - p0, seg) / denom)
        return int(round(np.clip(t, 0, 1) * (N_FRAMES - 1)))

    left0, right0, _ = pose_to_fingers(start)
    left1, right1, _ = pose_to_fingers(end)
    hit_frame_l = hit_frame(left0, left1, result["left"])
    hit_frame_r = hit_frame(right0, right1, result["right"])

    (center_line,) = ax.plot([], [], [], color="black", lw=2, label="center")
    (left_line,) = ax.plot([], [], [], color="tab:blue", lw=2, label="left finger")
    (right_line,) = ax.plot([], [], [], color="tab:orange", lw=2, label="right finger")
    center_pt = ax.scatter([], [], [], color="black", s=30)
    left_pt = ax.scatter([], [], [], color="tab:blue", s=30)
    right_pt = ax.scatter([], [], [], color="tab:orange", s=30)
    ax.legend(loc="upper right", fontsize=8)

    for i in range(N_FRAMES):
        # A finger held at p0 (blocked, no top-crossing point) never advances
        # past frame 0; a top-crossing finger advances normally and turns
        # red once it passes the computed hit frame.
        li = 0 if hit_frame_l == 0 else i
        ri = 0 if hit_frame_r == 0 else i

        center_line.set_data_3d(centers[: i + 1, 0], centers[: i + 1, 1], centers[: i + 1, 2])
        left_line.set_data_3d(lefts[: li + 1, 0], lefts[: li + 1, 1], lefts[: li + 1, 2])
        right_line.set_data_3d(rights[: ri + 1, 0], rights[: ri + 1, 1], rights[: ri + 1, 2])

        left_color = "red" if hit_frame_l is not None and i >= hit_frame_l else "tab:blue"
        right_color = "red" if hit_frame_r is not None and i >= hit_frame_r else "tab:orange"
        left_line.set_color(left_color)
        right_line.set_color(right_color)

        center_pt._offsets3d = (centers[i : i + 1, 0], centers[i : i + 1, 1], centers[i : i + 1, 2])
        left_pt._offsets3d = (lefts[li : li + 1, 0], lefts[li : li + 1, 1], lefts[li : li + 1, 2])
        right_pt._offsets3d = (rights[ri : ri + 1, 0], rights[ri : ri + 1, 1], rights[ri : ri + 1, 2])
        left_pt.set_color(left_color)
        right_pt.set_color(right_color)

        fig.canvas.draw_idle()
        if writer is not None:
            writer.grab_frame()
        else:
            plt.pause(FRAME_PAUSE)

    if writer is not None:
        for _ in range(HOLD_FRAMES):
            writer.grab_frame()
    else:
        plt.pause(HOLD_PAUSE)


if __name__ == "__main__":
    hd = build_synthetic_heightmap()
    print(f"heightmap: grid={hd.hmap.shape}, origin={hd.origin}, cell={hd.cell_size}, "
          f"occupied={(hd.hmap > 0).sum()} cells, workspace_bounds={WORKSPACE_BOUNDS}, "
          f"wall_margin={WALL_MARGIN}")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    writer = FFMpegWriter(fps=VIDEO_FPS)
    with writer.saving(fig, VIDEO_PATH, dpi=100):
        for scenario in build_scenarios():
            result = evaluate_scenario(hd, scenario)
            animate_scenario(fig, ax, hd, scenario, result, writer)

    plt.close(fig)
    print(f"\nSaved video to {os.path.abspath(VIDEO_PATH)}")
