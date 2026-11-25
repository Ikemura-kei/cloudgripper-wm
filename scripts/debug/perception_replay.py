# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["sam2>=1.1.0", "torch", "torchvision", "numpy", "opencv-python-headless", "huggingface_hub", "pyyaml", "scikit-learn"]
# ///
"""Replay recorded camera sweeps through SceneTracker, as if they came live from the cell. No robot needed.

  uv run scripts/debug/perception_replay.py SWEEP_DIR [SWEEP_DIR ...] [--top-every 1] [--out replay.mp4]

A sweep dir holds base_NNNN.jpg (undistorted) and top_NNNN.jpg (written by the parent repo's
scripts/cloudgripper_capture_sweep.py). Several dirs are played one after the other as one stream, with a refresh()
at each join, which stands in for a parking pose after the objects were rearranged. Prints each object's status after
start() and every refresh(), the step latency, and writes a side-by-side video with the masks.

If a dir's frames.jsonl has the commanded x, y per frame (cmd_xy), every move is also put through the collision
filter (cloudgripper_wm/collision_filter.py) against the objects as seen one frame earlier: its verdict is
printed when it changes and the finger tips it assumes are drawn on the base view, at the commanded target of the
move, not the measured pose (green: touches nothing, yellow: pushes a free object, red: blocked). The recorded arm moved regardless, so this shows where the filter predicts contact, not what it prevents."""
import argparse
import glob
import json
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")  # SAM 2 warns on every call that its optional hole-filling extension is not built

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cloudgripper_wm import collision_filter  # noqa: E402
from cloudgripper_wm.collision_filter import check  # noqa: E402
from cloudgripper_wm.perception import INVENTORY, SceneTracker, draw_states  # noqa: E402

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("sweeps", nargs="+")
p.add_argument("--top-every", type=int, default=1, help="update the top view every N steps (0: only at start/refresh)")
p.add_argument("--out", default=None, help="video path (default: replay.mp4 in the last sweep dir)")
p.add_argument("--one-model", action="store_true", help="use the small model for the promptless pass too")
p.add_argument("--image-size", type=int, default=768, help="tracker's internal resolution")
p.add_argument("--finger-sep-mm", type=float, default=None,
               help="finger tip spacing for the filter (default: the open gripper, 23; the 2026-11-23 recordings had it closed, 7)")
p.add_argument("--save-masks", default=None, help="write every step's masks to this .npz")
a = p.parse_args()
def report(tag, states, dt):
    print(f"{tag} ({dt * 1e3:.0f} ms): " + "; ".join(
        f"{n} {s.status}" + (f" on {s.on}" if s.on else "") + (f" at ({s.xy[0]:.2f},{s.xy[1]:.2f})" if s.xy else "")
        for n, s in states.items()), flush=True)


if a.finger_sep_mm:
    collision_filter.FINGER_SEP_MM = a.finger_sep_mm
tracker = SceneTracker(proposal_model=None if a.one_model else "facebook/sam2.1-hiera-large", image_size=a.image_size)
tmp = tempfile.mkdtemp()
lat, k, saved = [], 0, []
for si, sweep in enumerate(a.sweeps):
    bases, tops = sorted(glob.glob(sweep + "/base_*.jpg")), sorted(glob.glob(sweep + "/top_*.jpg"))
    log = Path(sweep, "frames.jsonl")
    cmd = [json.loads(l).get("cmd_xy") for l in log.read_text().splitlines()] if log.exists() else []
    said = None
    for fi, (fb, ft) in enumerate(zip(bases, tops)):
        base, top = cv2.imread(fb), cv2.imread(ft)
        verdict = None
        if 0 < fi < len(cmd) and cmd[fi] and cmd[fi - 1] and k > 0:
            # height, rotation and gripper are not in the log; the sweeps hold them at z 0.45, rotation 0
            verdict = check([*cmd[fi - 1], 0.45, 0.0, 1.0], [*cmd[fi], 0.45, 0.0, 1.0], states)
            if verdict.reason != said:
                said = verdict.reason
                print(f"{Path(sweep).name} frame {fi} cmd {cmd[fi][0]:.3f},{cmd[fi][1]:.3f}: filter says {said}", flush=True)
        torch.cuda.synchronize(); t = time.perf_counter()
        if k == 0:
            states = tracker.start(base, top); kind = "start"
        elif fi == 0:
            states = tracker.refresh(base, top); kind = "refresh"
        else:
            use_top = a.top_every > 0 and k % a.top_every == 0
            states = tracker.step(base, top if use_top else None); kind = "step2" if use_top else "step1"
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        if kind in ("start", "refresh"):
            report(f"{Path(sweep).name} {kind}", states, dt)
        else:
            lat.append((kind, dt))
        label = f"{Path(sweep).name} frame {fi} ({kind})"
        if verdict:
            colour = (0, 0, 255) if not verdict.allowed else (0, 255, 255) if verdict.pushed else (0, 255, 0)
            for poly in verdict.fingers:
                cv2.polylines(base, [np.int32([tracker.conv.x_y_to_px_py(x, y) for x, y in poly])], True, colour, 2)
            what = "BLOCKED" if not verdict.allowed else "PUSH (allowed)" if verdict.pushed else "FREE"
            label = f"filter: {what}: {verdict.reason}   |   boxes = finger tips ({collision_filter.FINGER_SEP_MM:g} mm apart) at the commanded target   |   {label}"
        cv2.imwrite(f"{tmp}/{k:05d}.jpg", draw_states(base, top, states, label))
        saved.append(np.stack([np.zeros((480, 640), bool) if states[n].mask_base is None else states[n].mask_base for n in INVENTORY]))
        k += 1
    report(f"{Path(sweep).name} last frame", states, 0)
for kind, what in (("step1", "base view only"), ("step2", "both views")):
    v = np.array([d for q, d in lat if q == kind]) * 1e3
    if len(v):
        print(f"step, {what}: median {np.median(v):.0f} ms, 95th percentile {np.percentile(v, 95):.0f} ms, max {v.max():.0f} ms ({len(v)} steps)")
print(f"GPU memory: {torch.cuda.memory_allocated() / 2**30:.2f} GB now, {torch.cuda.max_memory_allocated() / 2**30:.2f} GB peak; missing at the end: "
      f"{[n for n in INVENTORY if states[n].status == 'unaccounted']}")
if a.save_masks:
    np.savez_compressed(a.save_masks, m=np.stack(saved), names=list(INVENTORY))
out = a.out or str(Path(a.sweeps[-1]) / "replay.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "5", "-i", f"{tmp}/%05d.jpg", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "22", out], check=True)
print("video:", out)
