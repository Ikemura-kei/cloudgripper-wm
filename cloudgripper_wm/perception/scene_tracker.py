"""Object perception for ONE specific CloudGripper setup: cell 23 with its five known objects.

Read README.md in this folder first: it lists every assumption baked in here. This is not a general detector.

    tracker = SceneTracker("robot23")
    states = tracker.start(base_bgr, top_bgr)        # at a parking pose: find and name the objects (about 1 s)
    states = tracker.step(base_bgr)                  # every control step, base view only (about 45 ms)
    states = tracker.step(base_bgr, top_bgr)         # also update the top view (about twice the time)
    states = tracker.refresh(base_bgr, top_bgr)      # at a parking pose: look again for objects not on the glass

`base_bgr` is the UNDISTORTED base image (640x480), `top_bgr` the top image (1280x720). `states` maps each
inventory name to an ObjectState.
"""

from __future__ import annotations

import contextlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------------------------------------------------
# What we know about the setup (see README.md). Sizes are in pixels of the undistorted base view of cell 23, measured
# from SAM 2 masks on 2026-11-23; the cubes look larger than their 10 mm footprint because the camera also sees
# their sides.
# ---------------------------------------------------------------------------------------------------------------------
RIGID = {  # kind: (short side px, long side px, solidity, height mm)
    "cube": (34.5, 36.0, 0.96, 10.0),
    "slot": (56.0, 86.0, 0.945, 16.0),
    "pan": (69.0, 102.0, 0.905, 7.0),
}
INVENTORY = ("cube_blue", "cube_white", "slot", "pan", "rope")
ROPE_HEIGHT_MM = 8.0
# The rope is one piece of known rough size (Kei, 2026-11-25): its mask is the largest connected region only, and a
# region larger than this is not the rope (seen: about 9,000 px from below and 18,000 px from the top camera; a mask
# that ran over the glass had 115,000 and more).
# ponytail: the largest piece only. When the arm or an object covers the rope mid-length the rest is dropped; merge
# pieces that line up when that matters.
ROPE_MAX_PX = dict(base=25_000, top=60_000)

# Colour prompts (Kei, 2026-11-25): each rigid object is found by its paint, in each view on its own, and SAM 2 is
# prompted with points inside that colour region. HSV ranges and areas measured on cell 20 on 2026-11-25 (one layout).
# ponytail: one set of numbers for today's paint, lamps and camera settings; another cell or repainting needs its own.
# Known limits: from below the white cube has the gantry rail's grey, and the blue cube the frame's and the cable
# chain's blue, so a cube touching those in the picture is not found by colour (the shape-based proposals take over).
COLOUR = {  # name: {view: [(hsv low, hsv high), ...]}
    "slot": dict(base=[((0, 100, 60), (6, 255, 150)), ((174, 100, 60), (179, 255, 150))],
                 top=[((0, 180, 150), (5, 255, 255)), ((175, 180, 150), (179, 255, 255))]),
    "pan": dict(base=[((80, 60, 25), (104, 255, 80))], top=[((78, 180, 70), (90, 255, 160))]),
    "cube_blue": dict(base=[((104, 90, 35), (118, 255, 90))], top=[((104, 200, 90), (114, 255, 150))]),
    "cube_white": dict(base=[((95, 20, 70), (115, 55, 130))], top=[((50, 0, 170), (80, 30, 255))]),
}
IN_NOTCH_FRAC = 0.8  # share of a cube's top-view mask inside the slot's outline for it to count as in the notch
AREA_PX = {"cube": (1050, 3200), "slot": (4300, 10500), "pan": (4300, 5900)}  # kind: (base view, top view)
AREA_TOL = (0.35, 2.5)  # a colour region or a mask outside these multiples of AREA_PX is not that object
SIZE_TOL, SOLIDITY_TOL = 0.15, 0.05  # how far a mask may be from its template

CELLS = {
    "robot23": dict(
        glass=(198, 56, 556, 421),  # x0, y0, x1, y1 of the glass interior in the undistorted base view
        # Object centres seen in both views (base px -> top px), from three layouts on 2026-11-23. The fitted homography
        # only has to drop a prompt on the object in the top view; it is not a metric calibration.
        base_to_top=(
            [(369, 164), (270, 219), (291, 281), (336, 371), (337, 131), (245, 388), (216, 247), (213, 288), (359, 174), (282, 325)],
            [(570, 410), (628, 378), (683, 388), (772, 425), (540, 395), (766, 374), (653, 357), (686, 356), (584, 415), (718, 384)],
        ),
    ),
    "robot20": dict(  # first layout, 2026-11-25: four object centres and the rope's two ends, read off one picture pair
        glass=(167, 72, 527, 437),
        rope_sat=(35, 90),  # rope 56-63 (median), bare glass 11-24
        base_to_top=(
            [(293, 140), (228, 190), (352, 196), (246, 266), (197, 326), (522, 205)],
            [(472, 425), (552, 390), (545, 480), (650, 400), (725, 382), (545, 700)],
        ),
    ),
}
PAD = 14  # the proposal crop includes this much of the frame, so an object against the wall is not cut
_CAM = Path(__file__).resolve().parents[1] / "camera_params"


@dataclass
class ObjectState:
    name: str
    status: str  # "on_glass" | "stacked" | "off_glass" | "unaccounted"
    on: str | None = None  # for "stacked": the object it sits on or in
    xy: tuple[float, float] | None = None  # robot x, y of the footprint centre (on_glass only)
    angle_deg: float | None = None  # in-plane angle of the fitted rectangle, mod 180 (mod 90 for a cube)
    footprint_xy: np.ndarray | None = None  # (N, 2) polygon in robot x, y (on_glass only)
    height_mm: float = 0.0  # own height; a stack's height is the sum over `on`
    inferred_px: tuple[float, float] | None = None  # stacked / off_glass: centre in the base view, inferred from the top view
    inferred_xy: tuple[float, float] | None = None  # the same in robot x, y (rough: an obstacle for the filter, not a measurement)
    inferred_angle_deg: float | None = None  # its in-plane angle in the base view, where known (a cube in the slot)
    in_notch: bool = False  # stacked on the slot and, in the top view, inside the slot's outline: in its notch
    mask_base: np.ndarray | None = None  # bool (480, 640), None if never prompted in this view
    mask_top: np.ndarray | None = None  # bool (720, 1280)


def one_piece(mask: np.ndarray, max_px: int) -> np.ndarray:
    """The mask's largest connected region; empty if there is none or it is larger than max_px."""
    k, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if k < 2:
        return np.zeros(mask.shape, bool)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == i) if stats[i, cv2.CC_STAT_AREA] <= max_px else np.zeros(mask.shape, bool)


def colour_region(hsv: np.ndarray, name: str, view: str, roi: np.ndarray) -> np.ndarray | None:
    """The one connected region inside roi that has the object's colour and about its size (a cube's must also be
    compact); None if there is none or more than one, so that a guess never becomes a prompt."""
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in COLOUR[name][view]:
        m |= cv2.inRange(hsv, lo, hi)
    m = cv2.morphologyEx(m & roi, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    kind, a = name.split("_")[0], AREA_PX[name.split("_")[0]][view == "top"]
    ok = [i for i in range(1, k) if AREA_TOL[0] * a <= stats[i, cv2.CC_STAT_AREA] <= AREA_TOL[1] * a]
    if kind == "cube":  # compact: about as wide as high, and filling its bounding box
        ok = [i for i in ok if 0.5 <= stats[i, 2] / stats[i, 3] <= 2 and stats[i, 4] >= 0.5 * stats[i, 2] * stats[i, 3]]
    if kind == "cube" and view == "base":
        ok = [i for i in ok if cube_like(lab == i)]
    return lab == ok[0] if len(ok) == 1 else None


def cube_like(mask: np.ndarray) -> bool:
    """Base view: is this one solid square of a cube's size? A cube measures 29 to 36 px a side at solidity 0.93 or
    more; the grey rail mount that was once taken for the white cube (2026-11-25) has solidity 0.54."""
    if not mask.any():
        return False
    short, long, solidity, _ = shape_features(mask)
    return solidity >= 0.8 and short >= 24 and long <= 44


def spread_points(region: np.ndarray, k: int) -> np.ndarray | None:
    """k points well inside the region, spread out by farthest-point sampling. The region is eroded by 3 px first:
    that removes thin same-coloured things attached to it (a wire behind the slot put a point off the slot)."""
    pts = np.argwhere(cv2.erode(region.astype(np.uint8), np.ones((7, 7), np.uint8)))[:, ::-1].astype(np.float32)
    if len(pts) < k:
        return None
    chosen = [pts[np.argmin(np.linalg.norm(pts - pts.mean(0), axis=1))]]  # start near the middle
    for _ in range(k - 1):
        d = np.min([np.linalg.norm(pts - c, axis=1) for c in chosen], axis=0)
        chosen.append(pts[int(d.argmax())])
    return np.array(chosen)


def shape_features(mask: np.ndarray) -> tuple[float, float, float, int]:
    """short side, long side (fitted rectangle), solidity (area / convex hull area), number of separate pieces."""
    cs = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    c = max(cs, key=cv2.contourArea)
    wh = cv2.minAreaRect(c)[1]
    return min(wh), max(wh), cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1.0), len(cs)


def assign_inventory(cands: list[dict], wanted: set[str]) -> dict[str, int]:
    """Give each wanted rigid object at most one candidate mask. A candidate is {short, long, solidity, pieces, hsv}.

    Shape proposes the kind and colour has to agree (it also tells the two cubes apart and rejects cube-sized grey
    robot parts). A slot or pan that fits no template is named by colour if it is the only such mask left.
    Returns {inventory name: candidate index}."""
    # ponytail: fixed colour gates for today's paint (slot red 2/170/110, pan green 85/90/50)
    gates = {"slot": lambda h, sat, val: (h <= 10 or h >= 170) and sat > 110, "pan": lambda h, sat, val: 70 <= h <= 100 and sat > 60}
    best: dict[str, tuple[float, int]] = {}
    for i, c in enumerate(cands):
        if c["pieces"] != 1:
            continue
        for kind, (s, l, so, _) in RIGID.items():
            d = max(abs(c["short"] - s) / s, abs(c["long"] - l) / l)
            if d >= SIZE_TOL or abs(c["solidity"] - so) >= SOLIDITY_TOL:
                continue
            name = kind
            if kind in gates and not gates[kind](*c["hsv"]):
                continue  # a slot on its side has about the pan's outline: the colour must agree
            if kind == "cube":
                h, sat, val = c["hsv"]
                # ponytail: fixed colour gates for today's paint and lighting (blue cube 110/110/57, white cube
                # 80/14/94, grey robot blocks S 12 V 41-50). If lighting varies across cells, confirm a cube across
                # two parking poses instead: robot parts move with the arm, objects do not.
                name = "cube_blue" if 95 <= h <= 130 and sat > 60 else "cube_white" if sat < 45 and val > 65 else None
            if name in wanted and (name not in best or d < best[name][0]):
                best[name] = (d, i)
    out = {n: i for n, (_, i) in best.items()}
    # A slot or pan that fits no template (on its side, tilted on the rope) is still in the cell: if exactly one
    # leftover mask has its colour and roughly its size, that is it.
    for name in sorted(wanted & gates.keys() - out.keys()):
        area = RIGID[name][0] * RIGID[name][1]
        left = [i for i, c in enumerate(cands) if i not in out.values() and c["pieces"] == 1 and gates[name](*c["hsv"])
                and 0.5 * area < c["short"] * c["long"] < 1.8 * area]
        if len(left) == 1:
            out[name] = left[0]
    return out


def _top_cube_name(hsv, in_container: bool = False) -> str | None:
    """Which cube a top-view colour is (blue cube 109/240/142, white cube 44/31/216 on 2026-11-23), or None.
    The blue frame in shade (114/218/100) is about the blue cube in shade (110/237/110, tilted in the slot,
    2026-11-26): darker blue counts as the cube only in or on a container, where the frame cannot be."""
    h, sat, val = hsv
    return "cube_blue" if 95 <= h <= 130 and sat > 120 and val > (80 if in_container else 110) else "cube_white" if sat < 70 and val > 150 else None


class _Stream:
    """SAM 2 video tracking fed one frame at a time. The predictor's state holds only the current frame, and results
    older than KEEP frames are dropped (the tracker looks back 7 frames), so memory stays flat on an endless stream."""

    KEEP = 32

    def __init__(self, predictor, first_bgr: np.ndarray):
        self.p, self.n, self.ids, self._next = predictor, 0, {}, 0
        dev = predictor.device
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
        self._std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
        with tempfile.TemporaryDirectory() as d:  # init_state only reads frames from disk
            cv2.imwrite(f"{d}/0.jpg", first_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
            self.st = predictor.init_state(video_path=d)
        self.st["images"] = {0: self._prep(first_bgr)}
        self.masks: dict[str, np.ndarray] = {}

    def _prep(self, bgr: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).to(self.p.device).permute(2, 0, 1).float() / 255
        x = torch.nn.functional.interpolate(x[None], (self.p.image_size,) * 2, mode="bilinear", antialias=True)[0]
        return (x - self._mean) / self._std

    def add(self, name: str, *, mask=None, box=None, points=None) -> None:
        """Prompt an object on the current frame; allowed at any time (new objects join the tracking set)."""
        if name not in self.ids:
            self.ids[name], self._next = self._next, self._next + 1
        i = self.ids[name]
        if mask is not None:
            self.p.add_new_mask(self.st, frame_idx=self.n, obj_id=i, mask=mask)
        else:
            pts = None if points is None else np.asarray(points, np.float32)
            self.p.add_new_points_or_box(self.st, frame_idx=self.n, obj_id=i, box=box, points=pts,
                                         labels=None if pts is None else np.ones(len(pts), np.int32))

    def remove(self, name: str) -> None:
        self.p.remove_object(self.st, self.ids.pop(name), need_output=False)
        self.masks.pop(name, None)

    def track(self, bgr: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Advance to a new frame (or, with None, re-run the current one after add) and return {name: bool mask}."""
        if not self.ids:
            return {}
        if bgr is not None:
            self.n += 1
            self.st["images"] = {self.n: self._prep(bgr)}
            self.st["num_frames"] = self.n + 1
        (_, ids, logits), = self.p.propagate_in_video(self.st, start_frame_idx=self.n, max_frame_num_to_track=0)
        m = (logits[:, 0] > 0).cpu().numpy()
        for o in self.st["output_dict_per_obj"].values():
            o["non_cond_frame_outputs"].pop(self.n - self.KEEP, None)
        for o in self.st["frames_tracked_per_obj"].values():
            o.pop(self.n - self.KEEP, None)
        names = {i: n for n, i in self.ids.items()}
        self.masks = {names[i]: m[k] for k, i in enumerate(ids)}
        return self.masks


class SceneTracker:
    def __init__(self, robot: str = "robot23", device: str = "cuda", model: str = "facebook/sam2.1-hiera-small",
                 proposal_model: str | None = "facebook/sam2.1-hiera-large", image_size: int = 768):
        """`proposal_model=None` uses `model` for the promptless pass too (one model in memory; fewer proposals).
        `image_size` is the tracker's internal resolution (SAM 2 was trained at 1024; 768 gives the same masks on this setup
        at 63 instead of 100 ms per two-view step, and 512 is no faster)."""
        import sam2.sam2_video_predictor as svp
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2_hf

        from cloudgripper_wm.utils.coordinate_converter import CoordinateConverter

        svp.tqdm = lambda it, **k: it  # the predictor prints a progress bar per call
        cell = CELLS[robot]  # KeyError: this cell has not been set up (README.md)
        self.x0, self.y0, self.x1, self.y1 = (v + s * PAD for v, s in zip(cell["glass"], (-1, -1, 1, 1)))
        self.rope_sat = cell.get("rope_sat", (20, 75))
        self.H = cv2.findHomography(np.float32(cell["base_to_top"][0]), np.float32(cell["base_to_top"][1]))[0]
        # where colour prompts may come from: the glass in the base view; in the top view the glass through the rough
        # base-to-top fit, widened (objects stand up from it, and the fit is rough)
        gx0, gy0, gx1, gy1 = cell["glass"]
        self.roi = dict(base=np.zeros((480, 640), np.uint8), top=np.zeros((720, 1280), np.uint8))
        self.roi["base"][gy0 + 3:gy1 - 3, gx0 + 3:gx1 - 3] = 255
        cv2.fillPoly(self.roi["top"], [np.int32(self._to_top([(gx0, gy0), (gx1, gy0), (gx1, gy1), (gx0, gy1)]))], 255)
        self.roi["top"] = cv2.dilate(self.roi["top"], np.ones((41, 41), np.uint8))
        self.colour_prompts = robot == "robot20"  # ponytail: the COLOUR numbers are cell 20's
        self.conv = CoordinateConverter(str(_CAM / f"cam-to-robot-points/camera-to-robot-cr{robot[5:]}.yaml"))
        self.pred = svp.SAM2VideoPredictor.from_pretrained(
            model, device=device, hydra_overrides_extra=[f"++model.image_size={image_size}"])
        prop = self.pred if proposal_model is None else build_sam2_hf(proposal_model, device=device)
        self.gen = SAM2AutomaticMaskGenerator(prop, points_per_side=32, min_mask_region_area=100)
        self.base: _Stream | None = None
        self.top: _Stream | None = None
        self.states: dict[str, ObjectState] = {}
        self._tried: dict[str, int] = {}  # frame of the last hand-off attempt per object

    # ------------------------------------------------------------------ public
    def start(self, base_bgr: np.ndarray, top_bgr: np.ndarray) -> dict[str, ObjectState]:
        with self._ctx():
            self.base, self.top = _Stream(self.pred, base_bgr), _Stream(self.pred, top_bgr)
        self.states = {n: ObjectState(n, "unaccounted") for n in INVENTORY}
        self._top_sat = cv2.cvtColor(top_bgr, cv2.COLOR_BGR2HSV)[..., 1]
        return self._look(base_bgr, top_bgr)

    def refresh(self, base_bgr: np.ndarray, top_bgr: np.ndarray) -> dict[str, ObjectState]:
        """Advance both views, then look again for every object that is not on the glass."""
        self.step(base_bgr, top_bgr)
        return self._look(base_bgr, top_bgr)

    def step(self, base_bgr: np.ndarray, top_bgr: np.ndarray | None = None) -> dict[str, ObjectState]:
        with self._ctx():
            self.base.track(base_bgr)
            if top_bgr is not None:
                self.top.track(top_bgr)
                self._top_sat = cv2.cvtColor(top_bgr, cv2.COLOR_BGR2HSV)[..., 1]
        self._update()
        if top_bgr is not None:
            self._handoff(base_bgr)
        return self.states

    def _handoff(self, base_bgr: np.ndarray) -> None:
        """A cube seen only in the top view and touching no other object has probably dropped onto the glass: prompt
        the base view where the top view says it is, and keep the result only if it looks like that cube. This is the
        fast path between two refresh() calls; untested on real drops before 2026-11-23's live run."""
        for name, s in self.states.items():
            if s.status != "off_glass" or not name.startswith("cube") or self.base.n - self._tried.get(name, -99) < 5:
                continue
            self._tried[name] = self.base.n
            ys, xs = np.nonzero(s.mask_top)
            px, py = cv2.perspectiveTransform(np.float32([[[xs.mean(), ys.mean()]]]), np.linalg.inv(self.H))[0, 0]
            if not (self.x0 <= px < self.x1 and self.y0 <= py < self.y1):
                continue
            with self._ctx():
                self.base.add(name, points=[(px, py)])
                m = self.base.track()[name]
            ok = False
            if m.sum() > 300:
                sh, lo, so, pieces = shape_features(m)
                hsv = tuple(np.median(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2HSV)[m], 0))
                ok = assign_inventory([dict(short=sh, long=lo, solidity=so, pieces=pieces, hsv=hsv)], {name}) == {name: 0}
            if not ok:
                self.base.remove(name)
        self._update()

    # ------------------------------------------------------------------ detection at a parking pose
    @contextlib.contextmanager
    def _ctx(self):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            yield

    def _propose(self, gen, bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> list[dict]:
        crop = bgr[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        with self._ctx():
            anns = gen.generate(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        out = []
        for a in sorted(anns, key=lambda a: -a["predicted_iou"]):
            m = np.zeros(bgr.shape[:2], bool)
            m[y0:y1, x0:x1] = a["segmentation"]
            s, l, so, pieces = shape_features(m)
            out.append(dict(mask=m, short=s, long=l, solidity=so, pieces=pieces, hsv=tuple(np.median(hsv[a["segmentation"]], 0))))
        return out

    def _to_top(self, pts) -> np.ndarray:
        return cv2.perspectiveTransform(np.float32(pts)[None], self.H)[0]

    def _look(self, base_bgr: np.ndarray, top_bgr: np.ndarray) -> dict[str, ObjectState]:
        missing = {n for n, s in self.states.items() if s.status != "on_glass"}
        rigid_missing = missing - {"rope"}
        for name in rigid_missing & self.base.ids.keys():  # prompted before but lost or rejected: prompt it afresh
            if not self.base.masks.get(name, np.zeros(1)).any():
                with self._ctx():
                    self.base.remove(name)
        # 1a. colour prompts: points inside the object's colour region, in each view on its own
        self.prompts = getattr(self, "prompts", {})  # (view, name) -> ("points" | "box" | "mask", value), for debugging
        if rigid_missing and self.colour_prompts:
            hb, ht = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2HSV), cv2.cvtColor(top_bgr, cv2.COLOR_BGR2HSV)
            for name in sorted(rigid_missing):
                rb, rt = colour_region(hb, name, "base", self.roi["base"]), colour_region(ht, name, "top", self.roi["top"])
                pb, pt = (None if r is None else spread_points(r, 3) for r in (rb, rt))
                with self._ctx():
                    if pb is not None and name not in self.base.ids:
                        self.base.add(name, points=pb)
                        self.prompts["base", name] = ("points", pb)
                    if pt is not None and name not in self.top.ids:
                        self.top.add(name, points=pt)
                        self.prompts["top", name] = ("points", pt)
                    elif rb is not None and name not in self.top.ids:  # no colour region up there: the old box
                        q = self._to_top(np.argwhere(rb)[:, ::-1])
                        (qx0, qy0), (qx1, qy1) = q.min(0), q.max(0)
                        box = np.float32([qx0 - 4, qy0 - 0.45 * (qy1 - qy0) - 4, qx1 + 4, qy1 + 4])
                        self.top.add(name, box=box)
                        self.prompts["top", name] = ("box", box)
        # 1b. what colour did not find in the base view: promptless masks on the glass, named by size and shape
        by_shape = {n for n in rigid_missing if n not in self.base.ids}
        if by_shape:
            taken = np.zeros(base_bgr.shape[:2], bool)
            for n, m in self.base.masks.items():
                if n not in missing:
                    taken |= m
            cands = [c for c in self._propose(self.gen, base_bgr, self.x0, self.y0, self.x1, self.y1)
                     if (c["mask"] & taken).sum() < 0.3 * c["mask"].sum()]
            for name, i in assign_inventory(cands, by_shape).items():
                m = cands[i]["mask"]
                with self._ctx():
                    self.base.add(name, mask=m)
                    if name not in self.top.ids:  # top view: a box around the footprint, stretched up for the height
                        q = self._to_top(np.argwhere(m)[:, ::-1])
                        (qx0, qy0), (qx1, qy1) = q.min(0), q.max(0)
                        self.top.add(name, box=np.float32([qx0 - 4, qy0 - 0.45 * (qy1 - qy0) - 4, qx1 + 4, qy1 + 4]))
        with self._ctx():
            self.base.track(), self.top.track()
        # 2. rope: a colour seed gives a few points, SAM 2 makes the mask
        if "rope" in missing and "rope" not in self.base.ids:
            pts = self._rope_points(base_bgr)
            if pts is not None:
                # top view: seeded by colour there too (Kei, 2026-11-25). Base points sent through the rough
                # base-to-top fit landed beside the rope on cell 20 and SAM 2 masked the glass; they are the fallback.
                top_pts = self._rope_points(top_bgr, top=True)
                with self._ctx():
                    self.base.add("rope", points=pts)
                    self.top.add("rope", points=top_pts if top_pts is not None else self._to_top(pts))
                    self.base.track(), self.top.track()
        self._update()
        # 3. top view: a cube still missing may sit on or in another object. The container's top-view mask (prompted with
        # a box) includes what it carries, so look inside it for a compact patch of another colour.
        top_hsv = cv2.cvtColor(top_bgr, cv2.COLOR_BGR2HSV)
        for n in [n for n in rigid_missing if n in self.top.ids and self.states[n].status != "on_glass"]:
            m = self.top.masks.get(n)  # a top-only track can drift onto something else: keep it only if it still looks right
            touching = [c for c in ("slot", "pan") if (cm := self.top.masks.get(c)) is not None and m is not None
                        and (cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & cm).any()]
            if m is None or m.sum() < 150 or _top_cube_name(np.median(top_hsv[m], 0), in_container=bool(touching)) != n:
                self.top.remove(n)
        hidden = {n for n in rigid_missing if n.startswith("cube") and self.states[n].status != "on_glass" and n not in self.top.ids}
        for cont in [n for n, s in self.states.items() if s.status == "on_glass" and n != "rope"] if hidden else []:
            cm = self.top.masks.get(cont)
            if cm is None or cm.sum() < 150:
                continue
            ch = np.median(top_hsv[cm], 0).astype(int)
            dh = np.abs(top_hsv[..., 0].astype(int) - ch[0])
            other = cm & ((np.minimum(dh, 180 - dh) > 15) | (np.abs(top_hsv[..., 1].astype(int) - ch[1]) > 80))
            other = cv2.morphologyEx(other.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            k, lab, stats, _ = cv2.connectedComponentsWithStats(other)
            for i in sorted(range(1, k), key=lambda i: -stats[i, cv2.CC_STAT_AREA]):
                x, y, w, h, area = stats[i]
                name = _top_cube_name(np.median(top_hsv[lab == i], 0), in_container=True)
                if area >= 250 and max(w, h) < 2.5 * min(w, h) and name in hidden:
                    with self._ctx():
                        self.top.add(name, box=np.float32([x - 3, y - 3, x + w + 3, y + h + 3]))
                    hidden.discard(name)
        with self._ctx():
            self.top.track()
        return self._update()

    def _rope_points(self, base_bgr: np.ndarray, k: int = 5, top: bool = False) -> np.ndarray | None:
        # ponytail: the rope is seeded by its lilac colour (hue 110-140, low saturation 20-75; the blue robot parts and
        # frame are saturated above 100). Another rope colour needs other numbers here. Cell 20's glass looks faintly
        # blue from below (saturation up to 24), so its floor is 35 (CELLS[...]["rope_sat"]).
        hsv = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2HSV)
        if top:
            # ponytail: top-view gate from cell 20 (rope hue 111-118, saturation 27-110; frame same hue but saturation
            # above 148; walls and white cube hue 55-87). Whole picture, no region of interest.
            m = cv2.morphologyEx(cv2.inRange(hsv, (109, 35, 75), (125, 130, 255)), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        else:
            m = cv2.inRange(hsv, (108, self.rope_sat[0], 70), (140, self.rope_sat[1], 255))
            roi = np.zeros_like(m)
            roi[self.y0 + PAD:self.y1 - PAD, self.x0 + PAD:self.x1 - PAD] = 255
            m = cv2.morphologyEx(m & roi, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n < 2 or stats[1:, cv2.CC_STAT_AREA].max() < 800:
            return None
        big = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]  # the largest patch only: one piece, see ROPE_MAX_PX
        pts = np.argwhere(cv2.erode(np.isin(lab, big).astype(np.uint8), np.ones((3, 3), np.uint8)))[:, ::-1].astype(np.float32)
        if len(pts) < k:
            return None
        chosen = [pts[len(pts) // 2]]  # farthest-point sampling: k points spread along the rope
        for _ in range(k - 1):
            d = np.min([np.linalg.norm(pts - c, axis=1) for c in chosen], axis=0)
            chosen.append(pts[int(d.argmax())])
        return np.array(chosen)

    def _px_to_xy(self, pts) -> np.ndarray:
        """(N, 2) base pixels -> (N, 2) robot x, y; CoordinateConverter.px_py_to_x_y for many points at once."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        fx, fy = self.conv.py_X_model, self.conv.px_Y_model
        x = fx["model"].predict(fx["poly_features"].transform(pts[:, 1:2]))[:, 0]
        y = fy["model"].predict(fy["poly_features"].transform(pts[:, 0:1]))[:, 0]
        return np.stack([x, y], 1)

    # ------------------------------------------------------------------ bookkeeping every step
    def _update(self) -> dict[str, ObjectState]:
        rt = self.top.masks.get("rope")
        if rt is not None and rt.any() and self._top_sat is not None:
            # The rope's top-view mask comes from a box prompt and can run onto the blue frame, also as one piece
            # with the rope: drop its pixels that are not rope-coloured, then the crumbs.
            # ponytail: fixed gate (the lilac rope's saturation is below 100, the frame's above 200); another rope
            # or frame colour needs another number
            rt = cv2.morphologyEx((rt & (self._top_sat < 140)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            self.top.masks["rope"] = rt.astype(bool)
        for view, stream in (("base", self.base), ("top", self.top)):
            for name, m in stream.masks.items():  # every object is one piece of about known size
                cap = ROPE_MAX_PX[view] if name == "rope" else AREA_TOL[1] * AREA_PX[name.split("_")[0]][view == "top"]
                if name != "rope":  # a rigid object has no thin parts: cut tails (a red wire behind the slot joined its mask)
                    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)).astype(bool)
                m = one_piece(m, cap)
                if view == "base" and name.startswith("cube") and not cube_like(m):
                    m = np.zeros_like(m)  # not a cube (a robot part, or the mask ran along the rail): not seen from below
                stream.masks[name] = m
        top_seen = {n: m for n, m in self.top.masks.items() if m.sum() >= 150}
        for name in INVENTORY:
            mb, mt = self.base.masks.get(name), self.top.masks.get(name)
            kind = name.split("_")[0]
            h = ROPE_HEIGHT_MM if name == "rope" else RIGID[kind][3]
            full = 2000 if name == "rope" else RIGID[kind][0] * RIGID[kind][1]
            s = ObjectState(name, "unaccounted", height_mm=h, mask_base=mb, mask_top=mt)
            if mb is not None and mb.sum() >= 0.3 * full:
                s.status = "on_glass"
                c = max(cv2.findContours(mb.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
                (cx, cy), _, ang = cv2.minAreaRect(c)
                poly = cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2)
                xy = self._px_to_xy(np.vstack([[cx, cy], poly]))
                s.xy, s.footprint_xy = (float(xy[0, 0]), float(xy[0, 1])), xy[1:]
                s.angle_deg = float(ang % (90 if kind == "cube" else 180))
            elif name in top_seen:
                near = cv2.dilate(top_seen[name].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
                under = {o: (near & m).sum() for o, m in top_seen.items() if o not in (name, "rope")}
                s.on = max(under, key=under.get) if under and max(under.values()) > 0 else None
                s.status = "stacked" if s.on else "off_glass"
                # ponytail: the homography is fitted on objects lying on the glass, so a raised cube lands off.
                # For a stacked cube, take its offset from the object under it in the top view and apply it to that
                # object's base-view centre, which cancels most of that. Good for a drawing and a rough obstacle only.
                c = [np.argwhere(top_seen[o])[:, ::-1].mean(0) for o in ([name, s.on] if s.on else [name])]
                b = cv2.perspectiveTransform(np.float32([c]), np.linalg.inv(self.H))[0]
                mo = self.base.masks.get(s.on) if s.on else None
                if s.on == "slot" and mo is not None and mo.any():
                    # a cube in the slot can only sit in its notch: the slot's centre, edges aligned with the slot's
                    (cx, cy), _, ang = cv2.minAreaRect(np.argwhere(mo)[:, ::-1].astype(np.float32))
                    s.inferred_px, s.inferred_angle_deg = (cx, cy), ang % 90
                    # in the notch, or only touching the slot from the side? In the top view a cube in the notch lies
                    # inside the slot's convex outline; one leaning against the slot mostly outside it
                    hull = cv2.fillConvexPoly(np.zeros(top_seen["slot"].shape, np.uint8), cv2.convexHull(np.argwhere(top_seen["slot"])[:, ::-1]), 1)
                    inside = (top_seen[name] & cv2.dilate(hull, np.ones((7, 7), np.uint8)).astype(bool)).sum()
                    s.in_notch = bool(inside >= IN_NOTCH_FRAC * top_seen[name].sum())
                elif mo is not None and mo.any():
                    p = b[0] - b[1] + np.argwhere(mo)[:, ::-1].mean(0)
                    # a cube on or in an object lies within its outline: pull the centre in where it does not
                    inner = np.argwhere(cv2.erode(mo.astype(np.uint8), np.ones((25, 25), np.uint8)))[:, ::-1]
                    s.inferred_px = tuple(inner[np.linalg.norm(inner - p, axis=1).argmin()].astype(float)) if len(inner) else tuple(p)
                else:
                    s.inferred_px = tuple(b[0])
                s.inferred_xy = tuple(float(v) for v in self._px_to_xy([s.inferred_px])[0])
            self.states[name] = s
        return self.states


COLOURS = dict(cube_blue=(255, 150, 50), cube_white=(0, 255, 255), slot=(0, 0, 255), pan=(0, 255, 0), rope=(255, 0, 255))


def draw_states(base_bgr: np.ndarray, top_bgr: np.ndarray, states: dict[str, ObjectState], label: str = "") -> np.ndarray:
    """Both views side by side with each object's mask outline (dashed square in the base view: a cube hidden from below, placed from the top view), over a black strip with the statuses and the label (2240x890)."""
    base, top = base_bgr.copy(), top_bgr.copy()
    for n, s in states.items():
        for img, m in ((base, s.mask_base), (top, s.mask_top)):
            if m is not None and m.any():
                cv2.drawContours(img, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, COLOURS[n], 2)
                inner = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8))  # a black line just inside: the colour alone is hard to see on some backgrounds
                cv2.drawContours(img, cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 0, 0), 1)
    for n, s in states.items():
        if s.inferred_px is not None:  # hidden from below: a dashed cube-sized square where the top view puts it
            r = (RIGID["cube"][0] + RIGID["cube"][1]) / 4
            t = np.deg2rad(s.inferred_angle_deg or 0.0)  # unknown angle: drawn upright
            sq = np.array(s.inferred_px) + r * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
            for a, b in zip(sq, sq[1:]):
                for t in np.arange(0, 1, 0.25):  # four dashes a side
                    cv2.line(base, tuple(np.int32(a + (b - a) * t)), tuple(np.int32(a + (b - a) * (t + 0.14))), COLOURS[n], 2)
    strip = np.zeros((170, 2240, 3), np.uint8)
    w = 2240 // len(states)
    for i, (n, s) in enumerate(states.items()):
        cv2.putText(strip, n, (w * i + 16, 42), 0, 1.2, COLOURS[n], 3, cv2.LINE_AA)
        cv2.putText(strip, s.status + (f" on {s.on}" if s.on else ""), (w * i + 16, 84), 0, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(strip, label, (16, 146), 0, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([np.hstack([cv2.resize(base, (960, 720)), top]), strip])
