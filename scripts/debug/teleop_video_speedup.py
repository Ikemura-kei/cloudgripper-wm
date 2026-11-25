# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless"]
# ///
"""Shorten a teleop_filter.py session video for showing: fast everywhere, real speed around blocked moves.

  uv run scripts/debug/teleop_video_speedup.py VIDEO.mp4 [--fast 8] [--pad 10]

Frames whose label line starts with BLOCKED, and --pad frames before and after, play at X1; all others at X--fast.
The current speed is written at the top right. Writes VIDEO_fast.mp4 (needs ffmpeg) and leaves VIDEO.mp4 alone.
Two passes over the file, one frame in memory at a time (a ten-minute session is 14 GB of frames)."""
import argparse
import os
import subprocess

import cv2
import numpy as np

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("video")
p.add_argument("--fast", type=int, default=8, help="speed-up of the frames away from blocked moves")
p.add_argument("--pad", type=int, default=10, help="frames kept at real speed before and after a blocked move")
a = p.parse_args()

# The label as draw_states writes it (font, size and place), to find it in a frame by template matching.
# ponytail: breaks if draw_states changes its label line; log the verdict per frame in teleop_filter.py then.
strip = np.zeros((170, 2240), np.uint8)
cv2.putText(strip, "BLOCKED:", (16, 146), 0, 1.2, 255, 2, cv2.LINE_AA)
tmpl = strip[106:164, 12:178]


def frames():
    c = cv2.VideoCapture(a.video)
    while True:
        ok, f = c.read()
        if not ok:
            return
        yield f


blocked = np.array([cv2.matchTemplate(cv2.cvtColor(f[826:884, 12:178], cv2.COLOR_BGR2GRAY), tmpl, cv2.TM_CCOEFF_NORMED)[0, 0] > 0.8
                    for f in frames()])
slow = np.convolve(blocked, np.ones(2 * a.pad + 1), "same") > 0
print(f"{len(blocked)} frames, {int(blocked.sum())} read BLOCKED in {int((np.diff(blocked.astype(int)) == 1).sum() + blocked[0])} "
      f"episodes, {int(slow.sum())} at X1", flush=True)
raw, out, v = a.video.replace(".mp4", "_raw.mp4"), a.video.replace(".mp4", "_fast.mp4"), None
for f, s in zip(frames(), slow):
    if v is None:
        v = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 4 * a.fast, f.shape[1::-1])  # sessions are recorded at 4 fps
    cv2.putText(f, "X1" if s else f"X{a.fast}", (f.shape[1] - 190, 80), cv2.FONT_HERSHEY_DUPLEX, 2.6, (0, 0, 0), 7, cv2.LINE_AA)
    for _ in range(a.fast if s else 1):
        v.write(f)
v.release()
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p", out], check=True)
os.remove(raw)
print(f"{out}: {(slow.sum() * a.fast + (~slow).sum()) / (4 * a.fast):.0f} s (was {len(slow) / 4:.0f} s)")
