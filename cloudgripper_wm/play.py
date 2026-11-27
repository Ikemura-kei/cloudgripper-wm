"""Unattended play-data collection on one cell (Kei, 2026-11-27). MOVES THE ROBOT.

    prim = Primitives(env, on_step=...)            # env: a FilteredCloudGripperWrapper after reset
    rec = Recorder(env, out)                       # prim's on_step must call rec.step
    reason = collect(prim, rec, out, episodes=20, rng=np.random.default_rng(0))

An episode is 3 to 6 primitives drawn by `Primitives.random_primitive` (fewer when the scene allows none); no new
primitive starts once MAX_STEPS steps are recorded (one under way finishes). Before each episode the scene check
runs; when it fails, `reset_objects` (it fixes the named objects, checking after each), else the collection stops for a person. The
self-check runs at the start, every CHECK_EVERY episodes and after a surprise (`Primitives.surprise`); a failed one
runs once more, and a second failure stops the collection. A stop raises the arm, opens the gripper and returns the
reason. Only the primitives' steps are recorded; parking, checks and resets between episodes are not.

The session folder `out`:
    session.jsonl            one line per event: episode start and end, each primitive and its result, scene checks,
                             resets, self-checks, the stop
    episodes/ep_NNNNN/       frames/NNNN_top.jpg, NNNN_base.jpg   the camera frames as taken (base: fisheye), before
                                                                   each step and after the last
                             steps.npz      state (commanded pose per frame), action (executed delta per step);
                                            written when the episode ends: a folder without it is incomplete
                             diag.jsonl     per step: label (primitive and phase), requested and executed action,
                                            the collision filter's verdict, held, commanded and reported pose
                             states/NNNN.npz  the tracker's states with masks per frame (`save_states`)
    self_checks/NNN_a/       each self-check's folder (`self_check`)
Training data is made from frames and steps.npz only (`scripts/data/play_to_lance.py`); the rest is diagnostics
(Kei, 2026-11-27: kept, apart from the training data).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

EPISODE_PRIMITIVES = (3, 6)  # Kei, 2026-11-27
MAX_STEPS = 300              # Kei: episodes of 100 to 300 steps
CHECK_EVERY = 15             # self-check every this many episodes (Kei)


def _json(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


class Recorder:
    """Records the steps of the primitives into an episode folder, between `start` and `end`; `step` does nothing
    outside them (parking, checks and resets are not recorded)."""

    def __init__(self, env, out) -> None:
        self.env, self.out, self.dir = env, Path(out), None

    def start(self, ep: int) -> None:
        self.dir = self.out / "episodes" / f"ep_{ep:05d}"
        (self.dir / "frames").mkdir(parents=True)
        (self.dir / "states").mkdir()
        self.state, self.action, self.n = [], [], 0
        self.diag = open(self.dir / "diag.jsonl", "w")
        self._frame()

    def _frame(self) -> None:
        from cloudgripper_wm.perception.scene_tracker import save_states  # here: that module loads torch
        u = self.env.unwrapped
        cv2.imwrite(str(self.dir / "frames" / f"{len(self.state):04d}_top.jpg"), u._last_top_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(self.dir / "frames" / f"{len(self.state):04d}_base.jpg"), u._last_base_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        save_states(self.dir / "states" / f"{len(self.state):04d}.npz", self.env.states)
        self.state.append(u._target_pos.copy())

    def step(self, label: str, blocked: bool) -> None:
        if self.dir is None:
            return
        e, u = self.env, self.env.unwrapped
        reported, _ = (u.robot.get_state_ws if u.use_ws else u.robot.get_state)()
        self.diag.write(json.dumps(dict(step=self.n, t=time.time(), label=label, requested=e.last_requested, executed=e.last_action,
                                        verdict=e.last_verdict, blocked=blocked, held=list(e.held), commanded=u._target_pos,
                                        reported=reported), default=_json) + "\n")
        self.action.append(e.last_action.copy())
        self.n += 1
        self._frame()

    def end(self) -> int:
        """Close the episode; returns its number of steps."""
        np.savez(self.dir / "steps.npz", state=np.array(self.state, np.float32), action=np.array(self.action, np.float32).reshape(-1, 5))
        self.diag.close()
        self.dir = None
        return self.n


def collect(prim, rec: Recorder, out, episodes: int, rng, check=None, check_every: int = CHECK_EVERY) -> str:
    """Run `episodes` play episodes into `out` (continuing the episode numbers already there). `check(prim, out=DIR)`
    is the self-check (default `self_check.self_check`, against the robot's reference in git; DIR keeps its
    pictures and log); it returns (status, lines). Returns "done" or the reason it stopped."""
    if check is None:
        from cloudgripper_wm.self_check import self_check as check
    out = Path(out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)

    def log(event: str, **kw) -> None:
        with open(out / "session.jsonl", "a") as f:
            f.write(json.dumps(dict(t=time.time(), event=event, **kw), default=_json) + "\n")
        print(f"[play] {event} {kw}", flush=True)

    def stop(reason: str) -> str:
        prim.goto("stop: up", z=1.0)
        prim.goto("stop: open", grip=prim.open)
        log("stop", reason=reason)
        return reason

    def self_checked(tag: str) -> bool:
        n = len(list((out / "self_checks").glob("*_a"))) if (out / "self_checks").exists() else 0
        for attempt in "ab":
            status, lines = check(prim, out=out / "self_checks" / f"{n:03d}_{attempt}")  # by name: the 2nd parameter is the reference
            log("self_check", tag=tag, attempt=attempt, status=status, report=lines)
            if status == "pass":
                return True
        return False

    first = len(list((out / "episodes").glob("ep_*")))
    log("session", episodes=episodes, first=first)
    due = "start"
    for ep in range(first, first + episodes):
        if due:
            if not self_checked(due):
                return stop(f"self-check failed twice ({due})")
            due = None
        problems = prim.scene_check()
        if problems:
            log("scene_check", episode=ep, problems=problems)
            problems = prim.reset_objects()  # fixes the named objects, checking the scene after each
            log("reset", episode=ep, problems_left=problems)
            if problems:
                return stop("scene not fit after a reset: " + "; ".join(problems))
        rec.start(ep)
        k = int(rng.integers(EPISODE_PRIMITIVES[0], EPISODE_PRIMITIVES[1] + 1))
        log("episode", episode=ep, primitives=k)
        done = 0
        while done < k and rec.n < MAX_STEPS:
            pick = prim.random_primitive(rng)
            if pick is None:
                log("primitive", episode=ep, none="the scene allows no primitive")
                break
            name, args = pick
            prim.surprise = None
            ok = getattr(prim, name)(*args)
            log("primitive", episode=ep, name=name, args=args, ok=ok, steps=rec.n, surprise=prim.surprise)
            done += 1
            if prim.surprise:
                due = f"surprise: {prim.surprise}"
        log("episode_end", episode=ep, steps=rec.end(), primitives=done)
        if not due and (ep - first + 1) % check_every == 0:
            due = f"every {check_every} episodes"
    return "done"
