"""Demo: world-frame occupancy heightmap and 3D collision checking.

Loads debug_base.jpg (produced by test_connection.py) or falls back to a
synthetic mask. Builds a heightmap via the utility, checks three example
trajectories, and saves a 3D visualisation to occupancy_demo.png.

Usage
-----
    uv run python scripts/debug/occupancy_demo.py
"""

import time

import cv2
import numpy as np

from cloudgripper_wm.utils.occupancy import (
    HeightMapData,
    build_height_map,
    build_height_map_from_mask,
    check_collision,
)


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

class _FakeRobot:
    """Affine pixel→world map (metres). Counts calls to show efficiency."""
    def __init__(self, w, h, metres_per_px=0.001):
        self.cx, self.cy, self.s = w / 2.0, h / 2.0, metres_per_px
        self.calls = 0

    def px_py_to_x_y(self, px, py):
        self.calls += 1
        return ((px - self.cx) * self.s, (self.cy - py) * self.s)


def _synthetic_mask() -> np.ndarray:
    m = np.zeros((360, 480), np.uint8)
    cv2.rectangle(m, (120, 110), (165, 155), 255, -1)
    cv2.rectangle(m, (250,  90), (285, 125), 255, -1)
    cv2.rectangle(m, (200, 230), (270, 300), 255, -1)
    return m


def visualize(hd: HeightMapData, height: float, trajectories: list,
              z_cells: int = 6, max_cols: int = 60, save_path: str | None = None):
    import matplotlib
    if save_path:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    hmap, origin, cell_size = hd.hmap, hd.origin, hd.cell_size
    rows, cols = hmap.shape

    if cols > max_cols:
        f = max_cols / float(cols)
        vrows = max(1, int(round(rows * f)))
        vh = cv2.resize(hmap, (max_cols, vrows), interpolation=cv2.INTER_NEAREST)
    else:
        vh = hmap
    vrows, vcols = vh.shape
    cs_x = cell_size * cols / vcols
    cs_y = cell_size * rows / vrows

    dz     = height / z_cells
    levels = np.ceil(vh / dz).astype(int)
    kk     = np.arange(z_cells)[None, None, :]
    filled = kk < levels.T[:, :, None]

    xc = origin[0] + np.arange(vcols + 1) * cs_x
    yc = origin[1] + np.arange(vrows + 1) * cs_y
    zc = np.linspace(0.0, height, z_cells + 1)
    xg, yg, zg = np.meshgrid(xc, yc, zc, indexing="ij")

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.voxels(xg, yg, zg, filled,
              facecolor=(0.0, 0.7, 0.2, 0.35), edgecolor=(0, 0.4, 0.1, 0.2))

    for tr in trajectories:
        s = np.asarray(tr["start"], float)
        e = np.asarray(tr["end"],   float)
        hit, _ = check_collision(tr["start"], tr["end"], hd,
                                 surface=tr.get("surface", "top"))
        color = "red" if hit else "tab:blue"
        ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color=color, lw=2.5,
                label=f'{tr.get("label","traj")} ({"HIT" if hit else "clear"})')
        ax.scatter([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color=color, s=20)

    xspan = vcols * cs_x
    yspan = vrows * cs_y
    ax.set_xlabel("X (world)"); ax.set_ylabel("Y (world)"); ax.set_zlabel("Z (world)")
    ax.set_zlim(0, height * 2)
    ax.set_box_aspect((xspan, yspan, max(xspan * 0.3, 1e-9)))
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"World occupancy (height={height:.4g})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120); plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    HEIGHT = 0.5
    CELL   = 0.006
    img_file = "/home/ikemura/Code/cloudgripper_wm/misc/captures/usy_20260610_144207_base.jpg"
    img = cv2.imread(img_file)
    if img is not None:
        h, w = img.shape[:2]
        robot = _FakeRobot(w, h, metres_per_px=0.001)
        t0 = time.time()
        hd = build_height_map(robot, img, cell_size=CELL, height=HEIGHT)
    else:
        print("debug_base.jpg not found → synthetic mask + _FakeRobot.")
        mask = _synthetic_mask()
        h, w = mask.shape
        robot = _FakeRobot(w, h, metres_per_px=0.001)
        t0 = time.time()
        hd = build_height_map_from_mask(robot, mask, cell_size=CELL, height=HEIGHT)

    rows, cols = hd.hmap.shape
    print(f"heightmap built in {time.time() - t0:.3f}s  "
          f"grid={rows}×{cols} @ {hd.cell_size}m/cell  "
          f"origin=({hd.origin[0]:.3f}, {hd.origin[1]:.3f})  "
          f"occupied={(hd.hmap > 0).sum()} cells  "
          f"px_py calls={robot.calls}")

    # example trajectories using a point over an occupied cell
    ys, xs = np.nonzero(hd.hmap > 0)
    mid = len(xs) // 2
    wx = hd.origin[0] + (xs[mid] + 0.5) * hd.cell_size
    wy = hd.origin[1] + (ys[mid] + 0.5) * hd.cell_size

    trajs = [
        {"start": (hd.origin[0], hd.origin[1], 0.05),
         "end":   (hd.origin[0] + cols * hd.cell_size,
                   hd.origin[1] + rows * hd.cell_size, 0.05),
         "label": "fly-over @5cm"},
        {"start": (wx, wy, 0.06), "end": (wx, wy, 0.0),
         "label": "descent onto top"},
        {"start": (hd.origin[0], wy, 0.02),
         "end":   (hd.origin[0] + cols * hd.cell_size, wy, 0.02),
         "label": "side push @2cm"},
    ]

    for tr in trajs:
        t0 = time.time()
        hit, pt = check_collision(tr["start"], tr["end"], hd)
        where = "none" if pt is None else "(%.3f, %.3f, %.3f)" % tuple(pt)
        print(f'  {tr["label"]:>20}: collision={hit}  hit={where}  '
              f'({(time.time()-t0)*1000:.1f} ms)')

    t0 = time.time()
    visualize(hd, HEIGHT, trajs, save_path=img_file.replace(".jpg", "_occupancy_demo.png"))
    print(f"visualization saved in {time.time()-t0:.3f}s → occupancy_demo.png")
