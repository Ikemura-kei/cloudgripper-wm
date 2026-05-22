"""Inspect a collected Lance dataset — visualize episodes as videos.

Controls:
    Space       pause / resume
    Right arrow next episode
    Left arrow  previous episode
    Q           quit

Usage:
    uv run python scripts/inspect_data.py data/test.lance
    uv run python scripts/inspect_data.py data/test.lance --save-dir /tmp/videos
"""

import argparse
import sys
from pathlib import Path

import cv2
import lance
import numpy as np


# Action dimension labels matching CloudGripperEnv action space
ACTION_LABELS = ["dx", "dy", "dz", "drot", "dgrip"]
STATE_LABELS  = [" x",  " y",  " z",  " rot",  " grip"]

DISPLAY_H = 320   # height each camera panel is scaled to
TOP_SCALE = 4     # pixels (64px) upscale factor for top camera


def decode_jpeg(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def make_text_panel(width: int, state: list[float], action: list[float]) -> np.ndarray:
    panel = np.zeros((160, width, 3), dtype=np.uint8)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    col_w = width // 2

    cv2.putText(panel, "STATE", (10, 20), font, 0.5, (200, 200, 200), 1)
    for i, (label, val) in enumerate(zip(STATE_LABELS, state)):
        cv2.putText(panel, f"{label}: {val:+.3f}", (10, 45 + i * 22), font, scale, (100, 220, 100), thick)

    cv2.putText(panel, "ACTION", (col_w + 10, 20), font, 0.5, (200, 200, 200), 1)
    for i, (label, val) in enumerate(zip(ACTION_LABELS, action)):
        cv2.putText(panel, f"{label}: {val:+.3f}", (col_w + 10, 45 + i * 22), font, scale, (100, 180, 255), thick)

    return panel


def build_frame(
    top_bgr: np.ndarray,
    base_bgr: np.ndarray,
    state: list[float],
    action: list[float],
    episode: int,
    step: int,
    total_steps: int,
) -> np.ndarray:
    # Scale top camera (64×64) to DISPLAY_H × DISPLAY_H
    top_disp = cv2.resize(top_bgr, (DISPLAY_H, DISPLAY_H), interpolation=cv2.INTER_NEAREST)

    # Scale base camera to DISPLAY_H height, preserving aspect ratio
    h, w = base_bgr.shape[:2]
    base_w = int(w * DISPLAY_H / h)
    base_disp = cv2.resize(base_bgr, (base_w, DISPLAY_H))

    total_w = DISPLAY_H + base_w
    cameras = np.hstack([top_disp, base_disp])

    # Header bar
    header = np.zeros((30, total_w, 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"Episode {episode}   Step {step + 1}/{total_steps}   "
        f"[SPACE] pause  [A/D] episode  [Q] quit",
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1,
    )

    # Camera labels
    cv2.putText(cameras, "top (64px)", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)
    cv2.putText(cameras, "base", (DISPLAY_H + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)

    text_panel = make_text_panel(total_w, state, action)
    return np.vstack([header, cameras, text_panel])


def load_episode(ds: lance.LanceDataset, episode_idx: int) -> list[dict]:
    table = ds.to_table(filter=f"episode_idx = {episode_idx}").to_pydict()
    n = len(table["step_idx"])
    order = sorted(range(n), key=lambda i: table["step_idx"][i])
    return [
        {
            "pixels":      table["pixels"][i],
            "pixels_base": table["pixels_base"][i],
            "state":       list(table["state"][i]),
            "action":      list(table["action"][i]),
            "step_idx":    table["step_idx"][i],
        }
        for i in order
    ]


def play_episode(
    ds: lance.LanceDataset,
    episode_idx: int,
    num_episodes: int,
    fps: int,
    save_dir: Path | None,
) -> str:
    """Play one episode. Returns 'next', 'prev', or 'quit'."""
    steps = load_episode(ds, episode_idx)
    writer = None
    paused = False
    step_i = 0
    delay_ms = max(1, 1000 // fps)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    while True:
        row = steps[step_i]
        top  = decode_jpeg(row["pixels"])
        base = decode_jpeg(row["pixels_base"])
        frame = build_frame(
            top, base,
            row["state"], row["action"],
            episode_idx, step_i, len(steps),
        )

        if save_dir is not None and writer is None:
            h, w = frame.shape[:2]
            path = str(save_dir / f"episode_{episode_idx:04d}.mp4")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            print(f"  Saving → {path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB for display and saving

        if writer is not None:
            writer.write(frame)
        
        cv2.imshow("CloudGripper data", frame)
        key = cv2.waitKey(1 if paused else delay_ms) & 0xFF

        if key == ord("q"):
            if writer:
                writer.release()
            return "quit"
        elif key == ord(" "):
            paused = not paused
        elif key == 83 or key == ord("d"):  # right arrow
            if writer:
                writer.release()
            return "next"
        elif key == 81 or key == ord("a"):  # left arrow
            if writer:
                writer.release()
            return "prev"

        if not paused:
            step_i += 1
            if step_i >= len(steps):
                step_i = 0  # loop episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to .lance dataset")
    parser.add_argument("--fps", type=int, default=4, help="Playback speed (frames per second)")
    parser.add_argument("--episode", type=int, default=0, help="Starting episode index")
    parser.add_argument("--save-dir", default=None, help="Save episodes as mp4 to this directory")
    args = parser.parse_args()

    ds = lance.dataset(args.dataset)
    episodes = sorted(set(ds.to_table(columns=["episode_idx"]).to_pydict()["episode_idx"]))
    print(f"Dataset: {args.dataset}")
    print(f"Episodes: {len(episodes)}  Rows: {ds.count_rows()}")

    save_dir = Path(args.save_dir) if args.save_dir else None
    ep_i = args.episode

    while True:
        ep = episodes[ep_i]
        result = play_episode(ds, ep, len(episodes), args.fps, save_dir)
        if result == "quit":
            break
        elif result == "next":
            ep_i = min(ep_i + 1, len(episodes) - 1)
        elif result == "prev":
            ep_i = max(ep_i - 1, 0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
