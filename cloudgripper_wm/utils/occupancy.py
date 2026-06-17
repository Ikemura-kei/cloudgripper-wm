"""Occupancy heightmap construction and 3D collision checking.

Two public functions:

    hmap_data = build_height_map(robot, bot_img, ...)
    collided, hit_point = check_collision(start, end, hmap_data, ...)

HeightMapData is a lightweight dataclass that bundles the grid array with
its world-frame metadata so callers never have to track origin / cell_size
separately.

All coordinates are in the same unit that ``robot.px_py_to_x_y`` returns
(metres if the calibration is metric, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Public data type
# ---------------------------------------------------------------------------

@dataclass
class HeightMapData:
    """Output of build_height_map — passed directly to check_collision."""
    hmap:       np.ndarray          # (rows, cols) float32, cell value = top height or 0
    origin:     tuple[float, float] # world (X, Y) of the grid's lower-left corner
    cell_size:  float               # side length of one grid cell (same unit as X/Y/Z)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_height_map_from_mask(
    robot,
    mask: np.ndarray,
    cell_size: float,
    height: float,
    *,
    poly_eps_px: float = 2.0,
    min_area_px: int = 20,
    pad_cells: int = 2,
    dilate_cells: int = 0,
) -> HeightMapData:
    """Like build_height_map but accepts a pre-built binary mask (uint8, 0/255)
    instead of a raw camera image. Useful when you already have a mask from
    a custom detector or a synthetic source."""
    hmap, origin, cs = _build_heightmap(
        mask, robot, cell_size, height, poly_eps_px, min_area_px, pad_cells, dilate_cells
    )
    return HeightMapData(hmap=hmap, origin=origin, cell_size=cs)


def build_height_map(
    robot,
    bot_img: np.ndarray,
    cell_size: float,
    height: float,
    *,
    # mask tuning
    hsv_lower: tuple[int, int, int] = (35, 60, 40),
    hsv_upper: tuple[int, int, int] = (90, 255, 255),
    open_ksize: int = 5,
    open_iter: int = 1,
    close_ksize: int = 7,
    # heightmap tuning
    poly_eps_px: float = 2.0,
    min_area_px: int = 20,
    pad_cells: int = 2,
    dilate_cells: int = 0,
) -> HeightMapData:
    """Build a world-frame occupancy heightmap from a base camera image.

    Parameters
    ----------
    robot       : object with ``px_py_to_x_y(px, py) -> (X, Y)``
    bot_img     : BGR base-camera image (H×W×3, uint8)
    cell_size   : side of one grid cell, in the same unit as px_py_to_x_y output
    height      : assumed object height (same unit); used to extrude the footprint

    Mask parameters
    ---------------
    hsv_lower/upper : HSV colour range for the objects (default: green)
    open_ksize/iter : erosion+dilation kernel and iteration count (removes speckle)
    close_ksize     : closing kernel size (fills small holes inside objects)

    Heightmap parameters
    --------------------
    poly_eps_px  : approxPolyDP tolerance — larger = fewer outline vertices (faster)
    min_area_px  : contours smaller than this are dropped
    pad_cells    : empty-cell border around the bounding box
    dilate_cells : inflate each footprint by N cells (gripper radius + margin)

    Returns
    -------
    HeightMapData   (pass directly to check_collision)
    """
    mask = _extract_mask(bot_img, hsv_lower, hsv_upper, open_ksize, open_iter, close_ksize)
    hmap, origin, cs = _build_heightmap(
        mask, robot, cell_size, height, poly_eps_px, min_area_px, pad_cells, dilate_cells
    )
    return HeightMapData(hmap=hmap, origin=origin, cell_size=cs)


def is_inside_occupancy(point: tuple[float, float, float] | np.ndarray, hmap_data: HeightMapData) -> bool:
    """Return True if `point` (X, Y, Z) is within an object's footprint and
    at or below its (assumed) top height.

    Used to catch the case where a finger has already entered an occupied
    column (e.g. via lateral movement, or because `height` overestimates the
    real object) — `check_collision(surface="top")` only fires on the
    initial crossing and won't detect further descent once already inside.
    """
    o = np.asarray(hmap_data.origin, float)
    p = np.asarray(point, float)
    rows, cols = hmap_data.hmap.shape

    xi = int(round((p[0] - o[0]) / hmap_data.cell_size))
    yi = int(round((p[1] - o[1]) / hmap_data.cell_size))
    if not (0 <= xi < cols and 0 <= yi < rows):
        return False

    top = hmap_data.hmap[yi, xi]
    return top > 0.0 and p[2] <= top


def object_near_wall(
    point_xy: tuple[float, float] | np.ndarray,
    hmap_data: HeightMapData,
    bounds: tuple[float, float],
    margin: float,
) -> str | None:
    """If (x, y) lands on an occupied heightmap cell, find the connected
    object (via connected-component labeling of the occupancy mask) that
    cell belongs to. If that object's bounding extent reaches within
    `margin` of a workspace edge, return which edge (``"x_min"``, ``"x_max"``,
    ``"y_min"``, ``"y_max"``); otherwise return None.

    Checking the whole connected object — not just the contact cell —
    matters because an object wider than `margin` can have its near-wall
    edge already jammed against the wall while the finger contacts it from
    the far side, well outside the margin band.

    Used to detect an object that has been pushed up against the workspace
    wall, so further pushes toward that edge can be blocked.
    """
    o = np.asarray(hmap_data.origin, float)
    x, y = float(point_xy[0]), float(point_xy[1])
    rows, cols = hmap_data.hmap.shape

    xi = int(round((x - o[0]) / hmap_data.cell_size))
    yi = int(round((y - o[1]) / hmap_data.cell_size))
    if not (0 <= xi < cols and 0 <= yi < rows):
        return None
    if hmap_data.hmap[yi, xi] <= 0.0:
        return None

    binary = (hmap_data.hmap > 0.0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    label = labels[yi, xi]
    ys, xs = np.where(labels == label)

    x_min = o[0] + xs.min() * hmap_data.cell_size
    x_max = o[0] + (xs.max() + 1) * hmap_data.cell_size
    y_min = o[1] + ys.min() * hmap_data.cell_size
    y_max = o[1] + (ys.max() + 1) * hmap_data.cell_size

    lo, hi = bounds
    if x_min - lo <= margin:
        return "x_min"
    if hi - x_max <= margin:
        return "x_max"
    if y_min - lo <= margin:
        return "y_min"
    if hi - y_max <= margin:
        return "y_max"
    return None


def check_collision(
    start: tuple[float, float, float] | np.ndarray,
    end:   tuple[float, float, float] | np.ndarray,
    hmap_data: HeightMapData,
    *,
    surface: str = "top",
    xy_step: float = 0.5,
    z_step: float | None = None,
    z_floor: float = 0.0,
) -> tuple[bool, np.ndarray | None]:
    """Test whether the straight segment start→end collides with the heightmap.

    Parameters
    ----------
    start, end  : 3-D world points (X, Y, Z) in the same unit as the heightmap
    hmap_data   : returned by build_height_map
    surface     : ``"top"``    — collide only when crossing the top face
                  ``"volume"`` — collide when entering the occupied volume from any side
    xy_step     : sampling step in grid cells along XY (smaller = more accurate)
    z_step      : sampling step in height units (auto if None)
    z_floor     : lower bound for "volume" mode

    Returns
    -------
    (collided: bool, first_hit_world: np.ndarray | None)
        first_hit_world is the approximate 3-D collision point, or None if clear.
    """
    return _segment_collision(
        hmap_data.hmap, start, end,
        origin=hmap_data.origin,
        cell_size=hmap_data.cell_size,
        surface=surface,
        xy_step=xy_step,
        z_step=z_step,
        z_floor=z_floor,
    )


# ---------------------------------------------------------------------------
# Internal implementation (unchanged from occupancy_demo.py)
# ---------------------------------------------------------------------------

def _extract_mask(
    img: np.ndarray,
    hsv_lower, hsv_upper,
    open_ksize: int, open_iter: int, close_ksize: int,
) -> np.ndarray:
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    k    = np.ones((open_ksize, open_ksize), np.uint8)
    mask = cv2.erode(mask,  k, iterations=open_iter)
    mask = cv2.dilate(mask, k, iterations=open_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((close_ksize, close_ksize), np.uint8))
    return mask


def _build_heightmap(
    mask, robot, cell_size, height,
    poly_eps_px, min_area_px, pad_cells, dilate_cells,
):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys_world = []
    for c in cnts:
        if cv2.contourArea(c) < min_area_px:
            continue
        if poly_eps_px > 0:
            c = cv2.approxPolyDP(c, poly_eps_px, True)
        verts = c.reshape(-1, 2)
        w = np.array([robot.px_py_to_x_y(int(px), int(py)) for px, py in verts],
                     dtype=float).reshape(-1, 2)
        if w.shape[0] >= 3:
            polys_world.append(w)
    if not polys_world:
        raise ValueError("No object outlines mapped to world space.")

    allp = np.vstack(polys_world)
    xmin, ymin = allp.min(0) - pad_cells * cell_size
    xmax, ymax = allp.max(0) + pad_cells * cell_size
    cols = max(1, int(np.ceil((xmax - xmin) / cell_size)))
    rows = max(1, int(np.ceil((ymax - ymin) / cell_size)))

    occ = np.zeros((rows, cols), np.uint8)
    for w in polys_world:
        ix = np.clip((w[:, 0] - xmin) / cell_size, 0, cols - 1)
        iy = np.clip((w[:, 1] - ymin) / cell_size, 0, rows - 1)
        poly = np.stack([ix, iy], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(occ, [poly], 255)

    if dilate_cells > 0:
        k = 2 * dilate_cells + 1
        occ = cv2.dilate(occ, np.ones((k, k), np.uint8))

    hmap = np.where(occ > 127, float(height), 0.0).astype(np.float32)
    return hmap, (float(xmin), float(ymin)), float(cell_size)


def _segment_collision(hmap, start, end, origin, cell_size,
                       surface, xy_step, z_step, z_floor):
    o  = np.asarray(origin, float)
    s  = np.asarray(start,  float)
    e  = np.asarray(end,    float)
    rows, cols = hmap.shape

    sg = np.array([(s[0]-o[0])/cell_size, (s[1]-o[1])/cell_size, s[2]])
    eg = np.array([(e[0]-o[0])/cell_size, (e[1]-o[1])/cell_size, e[2]])

    if z_step is None:
        top_max = float(hmap.max()) if hmap.size else 1.0
        z_step  = max(top_max / 10.0, 1e-9)

    dist_xy = np.hypot(eg[0]-sg[0], eg[1]-sg[1])
    dist_z  = abs(eg[2]-sg[2])
    n = max(2, int(np.ceil(max(dist_xy / xy_step, dist_z / z_step))) + 1)

    ts  = np.linspace(0.0, 1.0, n)
    g   = sg[None, :] * (1-ts)[:, None] + eg[None, :] * ts[:, None]
    xi  = np.round(g[:, 0]).astype(int)
    yi  = np.round(g[:, 1]).astype(int)
    z   = g[:, 2]

    in_b = (xi >= 0) & (xi < cols) & (yi >= 0) & (yi < rows)
    top  = np.zeros(n, float)
    top[in_b] = hmap[yi[in_b], xi[in_b]]
    occ  = in_b & (top > 0.0)

    def to_world(p):
        return np.array([o[0] + p[0]*cell_size, o[1] + p[1]*cell_size, p[2]])

    if surface == "volume":
        hit = occ & (z <= top) & (z >= z_floor)
        if hit.any():
            return True, to_world(g[int(np.argmax(hit))])
        return False, None

    delta = z - top
    pair  = occ[:-1] & occ[1:]
    s0, s1 = delta[:-1], delta[1:]
    cross = pair & (((s0 > 0) & (s1 <= 0)) | ((s0 <= 0) & (s1 > 0)))
    if cross.any():
        i = int(np.argmax(cross))
        denom = s0[i] - s1[i]
        tl = 0.0 if denom == 0 else s0[i] / denom
        return True, to_world(g[i] + tl * (g[i+1] - g[i]))
    return False, None
