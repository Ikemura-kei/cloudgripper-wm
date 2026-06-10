"""Animate gripper finger-tip trajectories against a world occupancy heightmap.

Builds a heightmap from debug_base.jpg (robot-normalized [0,1] X/Y, robot
z-scale height) and, for a handful of hand-picked (start pose -> end pose)
scenarios, computes the left/right finger-tip 3-D paths via get_finger_pos,
checks them against the heightmap, and animates the gripper moving from
start to end pose one scenario at a time.

Pose tuple = (x, y, z, gripper_width, rotation_norm), all in [0, 1] — matches
CloudGripperEnv's "state" convention. rotation_norm * 180deg is the physical
gripper yaw, which get_finger_pos expects in radians.

Usage
-----
    uv run python scripts/debug/gripper_collision_sim.py
"""

import os

import cv2

# cv2 points Qt at its own bundled (incompatible) plugins; drop that before
# matplotlib's Qt backend initializes, or figure creation fails.
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import numpy as np
import matplotlib.pyplot as plt

from cloudgripper_wm.utils.cloudgripper_image_processor import CloudGripperImageProcessor
from cloudgripper_wm.utils.occupancy import build_height_map, check_collision, HeightMapData
from cloudgripper_wm.utils.occupancy_viz import pose_to_fingers, voxel_grid


CAM_TO_ROBOT_YAML = "./cloudgripper_wm/camera_params/cam-to-robot-points/camera-to-robot-cr23.yaml"
CAM_PARAMS_YAML = "./cloudgripper_wm/camera_params/base-camera-calibration/camera-params-cr23.yaml"
BASE_IMG_PATH = "./debug_base.jpg"

HEIGHT = 0.35   # assumed object height, in robot z units [0, 1] (not metres)
CELL = 0.01     # heightmap grid cell size, in robot x/y units [0, 1]

N_FRAMES = 30
FRAME_PAUSE = 0.05
HOLD_PAUSE = 1.0


def build_scenarios(hd: HeightMapData) -> list[dict]:
    rows, cols = hd.hmap.shape
    occ_ys, occ_xs = np.nonzero(hd.hmap > 0)
    free_ys, free_xs = np.nonzero(hd.hmap == 0)

    def cell_to_world(ix, iy):
        return (hd.origin[0] + (ix + 0.5) * hd.cell_size,
                hd.origin[1] + (iy + 0.5) * hd.cell_size)

    ox, oy = cell_to_world(occ_xs[len(occ_xs) // 2], occ_ys[len(occ_ys) // 2])
    fx, fy = cell_to_world(free_xs[len(free_xs) // 2], free_ys[len(free_ys) // 2])
    clip = lambda v: float(np.clip(v, 0.0, 1.0))

    return [
        {
            "label": "high pass over object (expect clear)",
            "start": [clip(ox - 0.2), oy, 0.8, 1.0, 0.0],
            "end":   [clip(ox + 0.2), oy, 0.8, 1.0, 0.0],
        },
        {
            "label": "low pass through object (expect clear - side push allowed)",
            "start": [clip(ox - 0.2), oy, 0.1, 1.0, 0.0],
            "end":   [clip(ox + 0.2), oy, 0.1, 1.0, 0.0],
        },
        {
            "label": "descend onto object (expect collision)",
            "start": [ox, oy, 0.8, 1.0, 0.0],
            "end":   [ox, oy, 0.05, 1.0, 0.0],
        },
        {
            "label": "pass through free space (expect clear)",
            "start": [clip(fx - 0.15), fy, 0.1, 1.0, 0.0],
            "end":   [clip(fx + 0.15), fy, 0.1, 1.0, 0.0],
        },
        {
            "label": "open + rotate near object (expect clear - lateral contact only)",
            "start": [clip(ox - 0.05), oy, 0.15, 0.0, 0.0],
            "end":   [clip(ox - 0.05), oy, 0.15, 1.0, 0.5],
        },
    ]


def animate_scenario(hd: HeightMapData, scenario: dict) -> None:
    start = np.asarray(scenario["start"], dtype=float)
    end = np.asarray(scenario["end"], dtype=float)

    left0, right0, center0 = pose_to_fingers(start)
    left1, right1, center1 = pose_to_fingers(end)

    # Only the top face of each object is treated as an obstacle — lateral
    # contact (pushing an object from the side) is allowed.
    hit_l, pt_l = check_collision(left0, left1, hd, surface="top")
    hit_r, pt_r = check_collision(right0, right1, hd, surface="top")
    print(f"\n{scenario['label']}")
    print(f"  left  finger: collision={hit_l}  hit={pt_l}")
    print(f"  right finger: collision={hit_r}  hit={pt_r}")

    poses = start[None, :] * (1 - np.linspace(0, 1, N_FRAMES))[:, None] \
        + end[None, :] * np.linspace(0, 1, N_FRAMES)[:, None]
    lefts, rights, centers = zip(*(pose_to_fingers(p) for p in poses))
    lefts, rights, centers = np.array(lefts), np.array(rights), np.array(centers)

    def hit_frame(hit, pt, p0, p1):
        if not hit:
            return None
        seg = p1 - p0
        denom = float(np.dot(seg, seg))
        t = 0.0 if denom == 0 else float(np.dot(pt - p0, seg) / denom)
        return int(round(np.clip(t, 0, 1) * (N_FRAMES - 1)))

    hit_frame_l = hit_frame(hit_l, pt_l, left0, left1)
    hit_frame_r = hit_frame(hit_r, pt_r, right0, right1)

    xg, yg, zg, filled, xspan, yspan = voxel_grid(hd, HEIGHT, z_cells=3, grid_cells=25)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(xg, yg, zg, filled,
              facecolor=(0.0, 0.7, 0.2, 0.25), edgecolor=(0, 0.4, 0.1, 0.15))
    ax.set_xlabel("X (robot)"); ax.set_ylabel("Y (robot)"); ax.set_zlabel("Z (robot)")
    ax.set_zlim(0, HEIGHT * 2)
    ax.set_box_aspect((xspan, yspan, max(xspan * 0.3, 1e-9)))
    ax.set_title(scenario["label"])

    (center_line,) = ax.plot([], [], [], color="black", lw=2, label="center")
    (left_line,) = ax.plot([], [], [], color="tab:blue", lw=2, label="left finger")
    (right_line,) = ax.plot([], [], [], color="tab:orange", lw=2, label="right finger")
    center_pt = ax.scatter([], [], [], color="black", s=30)
    left_pt = ax.scatter([], [], [], color="tab:blue", s=30)
    right_pt = ax.scatter([], [], [], color="tab:orange", s=30)
    ax.legend(loc="upper right", fontsize=8)

    plt.ion()
    plt.show()
    for i in range(N_FRAMES):
        center_line.set_data_3d(centers[: i + 1, 0], centers[: i + 1, 1], centers[: i + 1, 2])
        left_line.set_data_3d(lefts[: i + 1, 0], lefts[: i + 1, 1], lefts[: i + 1, 2])
        right_line.set_data_3d(rights[: i + 1, 0], rights[: i + 1, 1], rights[: i + 1, 2])

        left_color = "red" if hit_frame_l is not None and i >= hit_frame_l else "tab:blue"
        right_color = "red" if hit_frame_r is not None and i >= hit_frame_r else "tab:orange"
        left_line.set_color(left_color)
        right_line.set_color(right_color)

        center_pt._offsets3d = (centers[i : i + 1, 0], centers[i : i + 1, 1], centers[i : i + 1, 2])
        left_pt._offsets3d = (lefts[i : i + 1, 0], lefts[i : i + 1, 1], lefts[i : i + 1, 2])
        right_pt._offsets3d = (rights[i : i + 1, 0], rights[i : i + 1, 1], rights[i : i + 1, 2])
        left_pt.set_color(left_color)
        right_pt.set_color(right_color)

        fig.canvas.draw_idle()
        plt.pause(FRAME_PAUSE)

    plt.pause(HOLD_PAUSE)
    plt.close(fig)
    plt.ioff()


if __name__ == "__main__":
    img = cv2.imread(BASE_IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"{BASE_IMG_PATH} not found — run test_connection.py first.")

    image_processor = CloudGripperImageProcessor(CAM_TO_ROBOT_YAML, CAM_PARAMS_YAML)
    hd = build_height_map(image_processor, img, cell_size=CELL, height=HEIGHT)
    print(f"heightmap: grid={hd.hmap.shape}, origin={hd.origin}, cell={hd.cell_size}, "
          f"occupied={(hd.hmap > 0).sum()} cells")

    for scenario in build_scenarios(hd):
        animate_scenario(hd, scenario)
