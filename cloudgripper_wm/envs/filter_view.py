"""Window and video of a FilteredCloudGripperWrapper session, for the supervised debug scripts."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from cloudgripper_wm import collision_filter
from cloudgripper_wm.perception import draw_states


class SessionView:
    def __init__(self, env, out: Path, window: bool = True, fps: int = 2) -> None:
        """Call show() after every env step. `out` gets check.mp4 when close() runs. window=False: video only."""
        self.env, self.out, self.window, self.fps, self.video = env, Path(out), window, fps, None
        self.out.mkdir(parents=True, exist_ok=True)

    def show(self, label: str, blocked: bool) -> None:
        u, tracker = self.env.unwrapped, self.env._tracker
        base = self.env._proc.undistort_fish_eye(u._last_base_img)
        if hasattr(tracker, "conv"):  # the tips as the filter assumes them, in the base view
            for poly in collision_filter.finger_polys(u._target_pos):
                cv2.polylines(base, [np.int32([tracker.conv.x_y_to_px_py(x, y) for x, y in poly])], True, (255, 255, 255), 2)
        pose = "x {:.3f} y {:.3f} z {:.2f} rot {:.0f} grip {:.2f}".format(*u._target_pos[:3], u._target_pos[3] * 180, u._target_pos[4])
        hold = f" holding {', '.join(self.env.held)}" if self.env.held else ""
        frame = draw_states(base, u._last_top_img, self.env.states, f"{label}: {'BLOCKED' if blocked else 'sent'}   |   at {pose}{hold}")
        if self.video is None:
            self.video = cv2.VideoWriter(str(self.out / "check_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, frame.shape[1::-1])
        self.video.write(frame)
        if self.window:
            cv2.imshow(f"filtered env, {u._robot_name}", cv2.resize(frame, None, fx=0.75, fy=0.75))
            cv2.waitKey(1)

    def close(self, save_pose: bool = True) -> None:
        """Finish the video (H.264, so that VS Code, browsers and chat apps play it) and save the final pose for
        teleop_filter.py, which can then start without --start."""
        if self.video is not None:
            self.video.release()
            raw = self.out / "check_raw.mp4"
            if subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(self.out / "check.mp4")]).returncode == 0:
                raw.unlink()
        if save_pose:
            u = self.env.unwrapped
            last = Path.home() / f".cache/cloudgripper_wm/{u._robot_name}_last_pose.json"
            last.parent.mkdir(parents=True, exist_ok=True)
            last.write_text(json.dumps([round(float(v), 4) for v in u._target_pos]))


def mock_scene() -> dict:
    """A made-up cell 20 scene for --mock runs of the debug scripts (no robot, no GPU)."""
    from types import SimpleNamespace

    def obj(x, y, w, h, l=None):
        half = np.array([w, l or w]) / 2 * collision_filter.MM
        return SimpleNamespace(status="on_glass", on=None, xy=(x, y), height_mm=h, mask_base=None, mask_top=None, inferred_px=None,
                               inferred_xy=None, footprint_xy=np.array([x, y]) + half * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]))
    rope = obj(0.5, 0.2, 60, 8, 8)
    rope.footprint_xy = np.array([[x, 0.2 + s * 4 * collision_filter.MM] for s in (-1, 1) for x in np.linspace(0.3, 0.7, 30)[::s]])
    return dict(cube_blue=obj(0.7, 0.34, 10, 10), cube_white=obj(0.1, 0.3, 10, 10), slot=obj(0.3, 0.55, 15, 16, 25.5), pan=obj(0.5, 0.7, 25, 7), rope=rope)
