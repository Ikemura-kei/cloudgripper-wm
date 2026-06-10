"""Teleoperation script for a single CloudGripper robot.

All camera display and keyboard input run on the main thread to avoid
Qt/OpenCV threading issues. Image refresh rate is ~2-3 fps (limited by
the robot HTTP API latency).

Controls
--------
  w / s       Y axis  +/-
  a / d       X axis  +/-
  r / f       Z axis  up / down
  q / e       Rotation +/-
  t / g       Gripper open / close
  h           Return to home position
  x / ESC     Quit

Usage
-----
    uv run python scripts/debug/teleop.py
    uv run python scripts/debug/teleop.py --robot robot5
    uv run python scripts/debug/teleop.py --robot robot23 --step 0.03
"""

import argparse
import os
import select
import sys
import termios
import tty

import cv2
import numpy as np
from client.cloudgripper_client import GripperRobot

from cloudgripper_wm.tasks.base import DEFAULT_HOME_POS


# ── tuneable defaults ─────────────────────────────────────────────────────────
DEFAULT_ROBOT  = 'robot23'
DEFAULT_STEP   = 0.05
GRIPPER_STEP   = 0.1
ROT_STEP       = 0.05
GRIPPER_MIN    = 0.2
GRIPPER_MAX    = 0.825
DISPLAY_H      = 400


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def getch(timeout: float = 0.0) -> str | None:
    """Return a pressed key if one is available within `timeout` seconds."""
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


def show_images(top_bgr, base_bgr, pos: list[float]) -> None:
    def resize_h(img, h):
        if img is None:
            return np.zeros((h, h, 3), dtype=np.uint8)
        ih, iw = img.shape[:2]
        return cv2.resize(img, (int(iw * h / ih), h))

    top_p  = resize_h(top_bgr,  DISPLAY_H)
    base_p = resize_h(base_bgr, DISPLAY_H)
    hud    = np.hstack([top_p, base_p])
    w      = hud.shape[1]

    cv2.putText(hud, "top",  (4, 22),              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,80), 1)
    cv2.putText(hud, "base", (top_p.shape[1]+4,22),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,80), 1)

    strip = np.zeros((50, w, 3), dtype=np.uint8)
    labels = ["x","y","z","rot","grip"]
    for i,(l,v) in enumerate(zip(labels, pos)):
        cv2.putText(strip, f"{l}={v:.3f}", (8 + i*(w//5), 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,220,100), 1)

    cv2.imshow("CloudGripper teleop", np.vstack([hud, strip]))
    cv2.waitKey(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', default=DEFAULT_ROBOT)
    parser.add_argument('--step',  type=float, default=DEFAULT_STEP)
    args = parser.parse_args()

    robot = GripperRobot(args.robot, os.environ['CLOUDGRIPPER_TOKEN'])
    step  = args.step

    pos = list(DEFAULT_HOME_POS.copy())
    print(f"Moving {args.robot} to home …")
    robot.step_action([pos[0], pos[1], pos[2], pos[3] * 180, pos[4]])
    print("w/s:y  a/d:x  r/f:z  q/e:rot  t/g:grip  h:home  x:quit\n")

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            # 1. fetch images and display (main thread — no threading issues)
            _, _, base_img, _, top_img, _ = robot.get_all_states()
            show_images(top_img, base_img, pos)

            # 2. non-blocking key check (returns immediately if no key)
            key = getch(timeout=0.0)
            if key is None:
                continue

            moved = False
            if   key == 'w':  pos[1] = clamp(pos[1] + step);                    moved = True
            elif key == 's':  pos[1] = clamp(pos[1] - step);                    moved = True
            elif key == 'a':  pos[0] = clamp(pos[0] - step);                    moved = True
            elif key == 'd':  pos[0] = clamp(pos[0] + step);                    moved = True
            elif key == 'r':  pos[2] = clamp(pos[2] + step);                    moved = True
            elif key == 'f':  pos[2] = clamp(pos[2] - step);                    moved = True
            elif key == 'q':  pos[3] = clamp(pos[3] + ROT_STEP);                moved = True
            elif key == 'e':  pos[3] = clamp(pos[3] - ROT_STEP);                moved = True
            elif key == 't':
                pos[4] = clamp(pos[4] + GRIPPER_STEP, GRIPPER_MIN, GRIPPER_MAX); moved = True
            elif key == 'g':
                pos[4] = clamp(pos[4] - GRIPPER_STEP, GRIPPER_MIN, GRIPPER_MAX); moved = True
            elif key == 'h':  pos = list(DEFAULT_HOME_POS.copy());               moved = True
            elif key in ('x', '\x1b', '\x03'):
                break

            if moved:
                robot.step_action([pos[0], pos[1], pos[2], pos[3] * 180, pos[4]])
                print(f"\r  x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}"
                      f"  rot={pos[3]*180:.1f}°  grip={pos[4]:.3f}    ", flush=True)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        cv2.destroyAllWindows()
        print("\nDone.")


if __name__ == '__main__':
    main()
