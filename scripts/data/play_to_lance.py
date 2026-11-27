# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["stable-worldmodel[data]", "opencv-python-headless", "numpy"]
# [tool.uv.sources]
# stable-worldmodel = { path = "../../../stable-worldmodel" }
# ///
"""Play sessions (scripts/data/collect_play.py) to a stable-worldmodel Lance dataset for training: per frame
`pixels` (top camera, RGB), `pixels_base` (base camera as taken, fisheye, RGB), `state` (commanded pose) and
`action` (the executed delta from that frame; NaN on an episode's last frame, as World.collect writes it). The
diagnostics (diag.jsonl, states/, self-checks) stay behind (Kei, 2026-11-27). Frames are kept at full size; training
resizes (scripts/train/lewm.py, img_size). Episodes without steps.npz (interrupted) are skipped.

  uv run scripts/data/play_to_lance.py OUT.lance SESSION [SESSION ...]

stable-worldmodel comes from the parent repo's checkout (lwm_benchmark/third_party/stable-worldmodel, the version the
benchmark trains with; this submodule's own pin is not checked out). Its `data` extra (lancedb) is not in the parent
venv, hence the script's own environment."""
import sys
from pathlib import Path

import cv2
import numpy as np


def episodes(sessions, count):
    for s in sessions:
        for ep in sorted(Path(s).glob("episodes/ep_*")):
            if not (ep / "steps.npz").exists():
                print(f"skipped (incomplete): {ep}", flush=True)
                continue
            z = np.load(ep / "steps.npz")
            n = len(z["state"])
            rgb = lambda f: cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            count[0], count[1] = count[0] + 1, count[1] + n
            yield dict(pixels=[rgb(ep / "frames" / f"{i:04d}_top.jpg") for i in range(n)],
                       pixels_base=[rgb(ep / "frames" / f"{i:04d}_base.jpg") for i in range(n)],
                       state=list(z["state"]),
                       action=list(z["action"]) + [np.full(5, np.nan, np.float32)])


def convert(out, sessions) -> None:
    from stable_worldmodel.data.format import get_format
    count = [0, 0]
    with get_format("lance").open_writer(out) as w:
        w.write_episodes(episodes(sessions, count))
    print(f"{count[0]} episodes, {count[1]} frames written to {out}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    convert(sys.argv[1], sys.argv[2:])
