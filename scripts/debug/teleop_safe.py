"""Teleop test for SafeCloudGripperWrapper.

Drives the robot through CloudGripperEnv + SafeCloudGripperWrapper so you can
verify the collision-avoidance logic interactively: drive the gripper toward
an object from above and confirm the wrapper holds position and prints a
warning instead of descending into it, while pushing it from the side is
allowed.

All camera display, the live occupancy/finger view, and keyboard input run on
the main thread to avoid Qt/OpenCV threading issues.

Controls
--------
Click on the camera window to give it keyboard focus, then:
  w / s       Y axis  +/-
  a / d       X axis  +/-
  r / f       Z axis  up / down
  q / e       Rotation +/-
  t / g       Gripper open / close
  h           Reset episode (back to home position)
  x / ESC     Quit

Usage
-----
    uv run python scripts/debug/teleop_safe.py
    uv run python scripts/debug/teleop_safe.py --robot robot23 --step 0.03
    uv run python scripts/debug/teleop_safe.py --use-mock   # no hardware needed
"""

import argparse

import numpy as np
import gymnasium as gym

import cloudgripper_wm.envs  # noqa: F401  registers env IDs
from cloudgripper_wm.envs.robot_pool import RobotPool
from cloudgripper_wm.envs.safe_cloudgripper_wrapper import SafeCloudGripperWrapper

# Imported after safe_cloudgripper_wrapper so its QT_QPA_PLATFORM_PLUGIN_PATH
# fixup (cv2 vs matplotlib Qt plugin conflict) has already run. cv2.imshow is
# avoided entirely here for the same reason — both image views use matplotlib
# so they share one Qt backend with the live occupancy view.
import matplotlib.pyplot as plt

# Several of our control keys (a, s, d, f, g, h, q, r) collide with
# matplotlib's default keyboard shortcuts (save, quit, grid, home, ...).
# Clear those bindings so key_press_event sees our keys instead.
for _km in (
    "keymap.fullscreen", "keymap.home", "keymap.back", "keymap.forward",
    "keymap.pan", "keymap.zoom", "keymap.save", "keymap.quit",
    "keymap.quit_all", "keymap.grid", "keymap.grid_minor",
    "keymap.yscale", "keymap.xscale", "keymap.all_axes", "keymap.help",
    "keymap.copy",
):
    if _km in plt.rcParams:
        plt.rcParams[_km] = []


DEFAULT_ROBOT = "robot23"
DEFAULT_STEP = 0.05
GRIPPER_STEP = 0.1
ROT_STEP = 0.05


class CameraView:
    """Live matplotlib view of the top + base camera images.

    Also captures keyboard input via matplotlib's key_press_event — click on
    this window, then press keys to drive the robot.
    """

    def __init__(self):
        self.fig, (self.ax_top, self.ax_base) = plt.subplots(1, 2, figsize=(8, 4))
        self._keys: list[str] = []
        self.fig.canvas.mpl_connect("key_press_event", lambda e: self._keys.append(e.key))
        plt.ion()
        plt.show(block=False)

    def pop_key(self) -> str | None:
        return self._keys.pop(0) if self._keys else None

    def update(self, top_rgb, base_rgb, pos) -> None:
        self.ax_top.cla()
        self.ax_base.cla()
        if top_rgb is not None:
            self.ax_top.imshow(top_rgb)
        if base_rgb is not None:
            self.ax_base.imshow(base_rgb)
        self.ax_top.set_title("top")
        self.ax_base.set_title("base")
        self.ax_top.axis("off")
        self.ax_base.axis("off")

        labels = ["x", "y", "z", "rot", "grip"]
        self.fig.suptitle("  ".join(f"{l}={v:.3f}" for l, v in zip(labels, pos)))
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self) -> None:
        plt.close(self.fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default=DEFAULT_ROBOT)
    parser.add_argument("--step", type=float, default=DEFAULT_STEP)
    parser.add_argument("--height", type=float, default=0.35, help="assumed object height (robot z units)")
    parser.add_argument("--cell-size", type=float, default=0.01, help="occupancy grid cell size")
    parser.add_argument("--use-mock", action="store_true", help="use GripperRobotMock (no hardware)")
    args = parser.parse_args()

    RobotPool.configure([args.robot])
    env = gym.make(
        "cloudgripper/Gripper-v0",
        max_delta=args.step,
        use_mock=args.use_mock,
    )
    env = SafeCloudGripperWrapper(env, cell_size=args.cell_size, height=args.height, live_view=True)
    cam_view = CameraView()

    obs, info = env.reset()
    print("Click on the camera window, then:")
    print("w/s:y  a/d:x  r/f:z  q/e:rot  t/g:grip  h:reset  x/esc:quit\n")

    try:
        while True:
            top_img = env.render()
            cam_view.update(top_img, info.get("pixels_base"), env.unwrapped._target_pos)

            key = cam_view.pop_key()
            if key is None:
                continue

            if key == "h":
                obs, info = env.reset()
                continue
            if key in ("x", "escape"):
                break

            action = np.zeros(5, dtype=np.float32)
            if   key == "w": action[1] = args.step
            elif key == "s": action[1] = -args.step
            elif key == "a": action[0] = -args.step
            elif key == "d": action[0] = args.step
            elif key == "r": action[2] = args.step
            elif key == "f": action[2] = -args.step
            elif key == "q": action[3] = ROT_STEP
            elif key == "e": action[3] = -ROT_STEP
            elif key == "t": action[4] = GRIPPER_STEP
            elif key == "g": action[4] = -GRIPPER_STEP
            else:
                continue

            obs, reward, terminated, truncated, info = env.step(action)
            print(f"\r  target_pos={env.unwrapped._target_pos}    ", flush=True)
            if terminated or truncated:
                print("\nepisode ended, resetting...")
                obs, info = env.reset()
    finally:
        env.close()
        cam_view.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
