"""Standalone test for SafeCloudGripperWrapper.push_objects_to_center().

Sweeps the gripper around all four edges of the workspace, pushing any
objects near the walls back toward the center. Useful for resetting the
workspace between data-collection episodes without manual intervention.

Usage
-----
    uv run python scripts/debug/reset_objects.py
    uv run python scripts/debug/reset_objects.py --robot robot23 --push-amount 0.2 --n-sweeps 7
    uv run python scripts/debug/reset_objects.py --use-mock   # no hardware needed
    uv run python scripts/debug/reset_objects.py --live-view  # show live 3D occupancy view + camera feed
    uv run python scripts/debug/reset_objects.py --live-view --video-dir misc/reset_videos
"""

import argparse
import os
from contextlib import ExitStack

import gymnasium as gym

import cloudgripper_wm.envs  # noqa: F401  registers env IDs
from cloudgripper_wm.envs.robot_pool import RobotPool
from cloudgripper_wm.envs.safe_cloudgripper_wrapper import SafeCloudGripperWrapper


DEFAULT_ROBOT = "robot23"


class CameraView:
    """Live matplotlib view of the top + base camera images."""

    def __init__(self, plt):
        self._plt = plt
        self.fig, (self.ax_top, self.ax_base) = plt.subplots(1, 2, figsize=(8, 4))
        plt.ion()
        plt.show(block=False)

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
        self._plt.pause(0.001)

    def close(self) -> None:
        self._plt.close(self.fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default=DEFAULT_ROBOT)
    parser.add_argument("--step", type=float, default=0.05, help="env max_delta")
    parser.add_argument("--push-amount", type=float, default=0.25)
    parser.add_argument("--n-sweeps", type=int, default=7, help="push positions per wall edge")
    parser.add_argument("--transit-z", type=float, default=0.5)
    parser.add_argument("--push-z", type=float, default=0.2)
    parser.add_argument("--dwell-time", type=float, default=1.0, help="dwell time during reset moves")
    parser.add_argument("--height", type=float, default=0.35, help="assumed object height (robot z units)")
    parser.add_argument("--cell-size", type=float, default=0.01, help="occupancy grid cell size")
    parser.add_argument("--grid-cells", type=int, default=25, help="live view voxel grid resolution per axis")
    parser.add_argument("--live-view", action="store_true", help="show live 3D occupancy view + camera feed")
    parser.add_argument(
        "--video-dir", default=None,
        help="if set (requires --live-view), save camera.mp4 + occupancy.mp4 here"
    )
    parser.add_argument("--video-fps", type=float, default=2.0, help="video frame rate (one frame per move)")
    parser.add_argument("--use-mock", action="store_true", help="use GripperRobotMock (no hardware)")
    args = parser.parse_args()

    if args.video_dir and not args.live_view:
        parser.error("--video-dir requires --live-view")

    RobotPool.configure([args.robot])
    env = gym.make(
        "cloudgripper/Gripper-v0",
        max_delta=args.step,
        use_mock=args.use_mock,
        max_episode_steps=100_000,
    )
    env = SafeCloudGripperWrapper(
        env,
        cell_size=args.cell_size,
        height=args.height,
        live_view=args.live_view,
        grid_cells=args.grid_cells,
    )

    cam_view = None
    cam_writer = None
    occ_writer = None
    if args.live_view:
        # SafeCloudGripperWrapper(live_view=True) already popped
        # QT_QPA_PLATFORM_PLUGIN_PATH and imported matplotlib (see
        # LiveOccupancyView) — this just reuses that import.
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        import matplotlib.pyplot as plt
        cam_view = CameraView(plt)

        if args.video_dir:
            from matplotlib.animation import FFMpegWriter
            os.makedirs(args.video_dir, exist_ok=True)
            cam_writer = FFMpegWriter(fps=args.video_fps)
            occ_writer = FFMpegWriter(fps=args.video_fps)

    def on_step(obs, info):
        if cam_view is not None:
            cam_view.update(env.render(), info.get("pixels_base"), env.unwrapped._target_pos)
        if cam_writer is not None:
            cam_writer.grab_frame()
        if occ_writer is not None:
            occ_writer.grab_frame()

    try:
        with ExitStack() as stack:
            if cam_writer is not None:
                cam_path = os.path.join(args.video_dir, "camera.mp4")
                stack.enter_context(cam_writer.saving(cam_view.fig, cam_path, dpi=100))
            if occ_writer is not None:
                occ_path = os.path.join(args.video_dir, "occupancy.mp4")
                stack.enter_context(occ_writer.saving(env._view.fig, occ_path, dpi=100))

            obs, info = env.reset()
            on_step(obs, info)
            env.push_objects_to_center(
                push_amount=args.push_amount,
                n_sweeps=args.n_sweeps,
                transit_z=args.transit_z,
                push_z=args.push_z,
                dwell_time=args.dwell_time,
                on_step=on_step,
            )
    finally:
        env.close()
        if cam_view is not None:
            cam_view.close()
        if args.video_dir:
            print(f"\nSaved videos to {os.path.abspath(args.video_dir)}")
        print("\nDone.")


if __name__ == "__main__":
    main()
