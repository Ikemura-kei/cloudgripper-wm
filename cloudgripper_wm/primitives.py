"""Scripted play primitives on the filtered env: push, reset to the centre, grasp and place, cube into the slot,
rope drag. THEY MOVE THE ROBOT.

    env = FilteredCloudGripperWrapper(CloudGripperEnv(...)); env.reset()
    prim = Primitives(env)
    prim.reset_objects()                       # push what is far from the centre back towards it
    prim.push("pan", (1, 0), 30)               # 30 mm along +x
    prim.pick_place("cube_white", (0.4, 0.5))
    prim.cube_into_slot("cube_blue")
    prim.rope_drag(0, (0.1, 0.0))              # grasp the rope near one end and drag it

Every primitive is a list of absolute moves. Each move is sent through `env.step` as delta actions within the env's
per-axis step limit (`constants.PLAY_MAX_DELTA`), the interface the world model uses, and the collision filter
judges each step. A primitive returns True if all its moves were sent. If the filter blocks one, the primitive is given up:
the arm goes to the top height and False is returned; the caller picks the next primitive (Kei, 2026-11-25).
Positions are read once, when the primitive starts, from `env.states`.

Written for cell 20 and its five objects. ponytail, all of it untested on the robot as of 2026-11-25:
- heights, approach distance and waiting times below are first guesses to tune in supervised runs;
- a primitive given up while holding an object drops it from the top height;
- taking a cube or the rope out of the slot, and the rope into the slot, are not written yet.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from cloudgripper_wm import collision_filter as cf

CENTRE = (0.44, 0.45)      # middle of cell 20's glass, robot x, y
CENTRE_RADIUS = 0.15       # reset: objects farther than this (21 mm) from CENTRE are pushed back to it
PUSH_TIP_MM = 3.0          # tip height while pushing an object that stands clear of the frame
OVER_FRAME_TIP_MM = 8.6    # ... and when the push starts over the frame (6.25 mm): only objects 10 mm and taller
APPROACH_MM = 10.0         # a push starts this far behind the object
SLOT_DROP_MM = 2.2         # a cube is let go with the tips this far above the slot's top
EDGE_SHIFT_MM = 2.0        # grasping a cube that lies against the frame: the tips go this far towards the glass
FREE_MM = 22.0             # reset: a cube is put down at least this far from every other object's outline


def z_for_tip(mm: float) -> float:
    """The height command that puts the tips `mm` above the glass (inverse of collision_filter.tip_mm)."""
    z, t = zip(*cf.Z_TO_TIP_MM)
    return float(np.interp(mm, t, z))


def _rect_angle(footprint_xy) -> tuple[float, float]:
    """Direction (rad) of the fitted rectangle's long side in robot x, y, and its long / short ratio."""
    (_, _), (w, h), a = cv2.minAreaRect(np.float32(footprint_xy) * 1000)
    return (np.deg2rad(a) if w >= h else np.deg2rad(a + 90)) % np.pi, max(w, h) / max(min(w, h), 1e-6)


class Primitives:
    def __init__(self, env, on_step: Callable[[str, bool], None] | None = None) -> None:
        """`env`: a FilteredCloudGripperWrapper after reset. `on_step(label, blocked)`: called after every move."""
        self.env, self.u, self.on_step = env, env.unwrapped, on_step
        self.open, self.closed = self.u._gripper_range[1], self.u._gripper_range[0]

    # ------------------------------------------------------------------ moves
    def goto(self, label: str, x=None, y=None, z=None, rot=None, grip=None) -> bool:
        """Move the named axes to absolute values, in equal delta steps within the env's per-axis limit, each
        through the filter; False as soon as one step is blocked."""
        target = np.array([v if v is not None else p for v, p in zip((x, y, z, rot, grip), self.u._target_pos)], np.float32)
        target[:2] = np.clip(target[0], *self.u._x_range), np.clip(target[1], *self.u._y_range)  # as the env will clip
        n = int(np.ceil((np.abs(target - self.u._target_pos) / self.u.max_delta).max() - 1e-6))
        for i in range(n, 0, -1):
            delta = (target - self.u._target_pos) / i  # what is left, in i equal steps: ends exactly on target
            # wait long enough for the arm to arrive before the next picture: the filter trusts the commanded pose
            wait, self.u.dwell_time = self.u.dwell_time, (0.5 + 4 * float(np.hypot(*delta[:2])) + 2 * abs(float(delta[2]))
                                                          + abs(float(delta[3])) + (0.5 if abs(delta[4]) > 1e-4 else 0)) * wait_scale(self.u)
            before = self.env.n_blocked
            try:
                self.env.step(delta)
            finally:
                self.u.dwell_time = wait
            blocked = self.env.n_blocked > before
            if self.on_step:
                self.on_step(label, blocked)
            if blocked:
                return False
        return True

    def _run(self, name: str, moves: list[tuple[str, dict]]) -> bool:
        for label, kw in moves:
            if not self.goto(f"{name}: {label}", **kw):
                self._give_up(name)
                return False
        return True

    def _give_up(self, name: str) -> None:
        """Arm to the top height; whatever is held is dropped from there, so the next primitive starts empty-handed."""
        self.goto(f"{name}: given up, raising", z=1.0)
        if self.env.held:
            self.goto(f"{name}: given up, dropping {', '.join(self.env.held)}", grip=self.open)

    def _let_go(self, name: str) -> bool:
        """Open the gripper here; if the filter blocks that (a tip would push a neighbour against the frame), go up
        a step at a time and open as soon as it is allowed: the object drops from there (Kei, 2026-11-25)."""
        while not self.goto(f"{name}: let go", grip=self.open):
            if self.u._target_pos[2] >= 1.0 or not self.goto(f"{name}: no room to open, up a step", z=min(1.0, float(self.u._target_pos[2]) + 0.25)):
                return False
        return True

    def _cmd(self, tips_xy, rot: float) -> dict:
        """The commanded x, y that puts the tips' midpoint at tips_xy with the gripper at `rot` (0 to 1)."""
        th = rot * np.pi
        turn = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ cf.TOOL_OFFSET_TURNING
        x, y = np.asarray(tips_xy, float) - cf.TOOL_OFFSET - turn
        return dict(x=float(x), y=float(y))

    def _on_glass(self, name: str):
        s = self.env.states.get(name)
        return s if s is not None and s.status == "on_glass" and s.footprint_xy is not None else None

    # ------------------------------------------------------------------ primitives
    def push(self, name: str, direction, dist_mm: float) -> bool:
        """Push `name` by dist_mm along `direction` (robot x, y) with the closed gripper held across the direction."""
        s = self._on_glass(name)
        if s is None:
            return False
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        c, fp = np.array(s.xy), np.asarray(s.footprint_xy)
        back = float(((c - fp) @ d).max())  # how far the object reaches behind its centre
        start = c - d * (back + (cf.FINGER_MM[1] / 2 + APPROACH_MM) * cf.MM)
        end = start + d * (APPROACH_MM + dist_mm) * cf.MM
        rot = ((np.arctan2(d[1], d[0]) + np.pi / 2) % np.pi) / np.pi
        x0, y0, x1, y1 = cf.GLASS
        m = 10 * cf.MM  # the closed tips reach 7 mm either side
        tip = PUSH_TIP_MM
        if not (x0 + m <= start[0] <= x1 - m and y0 + m <= start[1] <= y1 - m):  # the push starts over the frame
            if s.height_mm < 10:
                return False  # the tips cannot get low enough behind it (the pan at the frame)
            tip = OVER_FRAME_TIP_MM
        return self._run(f"push {name}", [("up", dict(z=1.0)), ("close", dict(grip=self.closed)), ("turn", dict(rot=rot)),
                                           ("behind it", self._cmd(start, rot)), ("down", dict(z=z_for_tip(tip))),
                                           ("push", self._cmd(end, rot)), ("up", dict(z=1.0))])

    def _free_spot(self, name: str) -> np.ndarray:
        """A place near CENTRE at least FREE_MM from every other object's outline (the nearest to CENTRE that is)."""
        others = [np.asarray(s.footprint_xy) for n, s in self.env.states.items() if n != name and s.footprint_xy is not None]
        ring = [np.array(CENTRE) + r * np.array([np.cos(t), np.sin(t)]) for r in (0, 0.07, 0.14) for t in np.arange(0, 2 * np.pi, np.pi / 4)]
        ok = [p for p in ring if all(np.linalg.norm(o - p, axis=1).min() > FREE_MM * cf.MM for o in others)]
        return ok[0] if ok else np.array(CENTRE)

    def reset_objects(self) -> dict[str, bool]:
        """Bring every object that is far from CENTRE back towards it, farthest first: a cube is picked up and put
        down on a free spot (that also works against the frame, where nothing can get behind it), the slot and
        the pan are pushed (a cube in a corner too), the rope's farther end is dragged. Returns {name: done} for what it tried.

        ponytail: a slot or pan lying against the low-x or low-y frame, and a pan against any frame, stay there
        (the push is given up): turn them with a sideways push first, or grasp the slot, when that happens often."""
        done = {}
        far = lambda n: float(np.linalg.norm(np.subtract(self.env.states[n].xy, CENTRE)))
        for name in sorted((n for n in self.env.states if n != "rope" and self._on_glass(n)), key=far, reverse=True):
            if far(name) <= CENTRE_RADIUS:  # read again: an earlier move may have moved it
                continue
            if name.startswith("cube"):
                done[name] = self.pick_place(name, self._free_spot(name)) or self.push(
                    name, np.subtract(CENTRE, self.env.states[name].xy), (far(name) - CENTRE_RADIUS) / cf.MM)  # a corner: push it out
            else:
                done[name] = self.push(name, np.subtract(CENTRE, self.env.states[name].xy), (far(name) - CENTRE_RADIUS) / cf.MM)
        if self._on_glass("rope"):
            ends = self._rope_ends()
            i = int(np.argmax([np.linalg.norm(e - CENTRE) for e, _ in ends]))
            to = np.subtract(CENTRE, ends[i][0])
            if np.linalg.norm(to) > CENTRE_RADIUS:
                done["rope"] = self.rope_drag(i, to * (1 - CENTRE_RADIUS / np.linalg.norm(to)))
        return done

    def grasp(self, name: str) -> bool:
        """Open, go down around `name` (a cube) with the tips along one of its sides, close, lift."""
        s = self._on_glass(name)
        if s is None:
            return False
        th, c = _rect_angle(s.footprint_xy)[0] % (np.pi / 2), np.array(s.xy, float)
        # Near the frame the tips go along it: the open tips reach 16 mm either way and would be over the frame
        # otherwise. Right against it they also go EDGE_SHIFT_MM towards the glass (the wide side of a tip would
        # be over the frame): they still hold 8 of the cube's 10 mm. In a corner neither direction fits.
        x0, y0, x1, y1 = cf.GLASS
        reach, near = 19 * cf.MM, 7 * cf.MM
        at_x, at_y = c[0] < x0 + reach or c[0] > x1 - reach, c[1] < y0 + reach or c[1] > y1 - reach
        ths = [th, th + np.pi / 2]  # the cube's two side directions
        if at_x != at_y:  # one frame side: the direction along that side first
            ths.sort(key=lambda t: abs(np.sin(t)) if at_x else abs(np.cos(t)), reverse=True)
        c += EDGE_SHIFT_MM * cf.MM * np.array([int(c[0] < x0 + near) - int(c[0] > x1 - near), int(c[1] < y0 + near) - int(c[1] > y1 - near)])
        # Ask the filter before moving: take the first direction in which the open tips can go down around the cube
        # (the other may put a tip on the rope or a neighbour). If neither can, give up without moving.
        for th in ths:
            rot = (th % np.pi) / np.pi
            over = [*self._cmd(c, rot).values(), 1.0, rot, self.open]
            verdict = cf.check(over, [*over[:2], 0.0, rot, self.open], self.env.states)
            if verdict.allowed:
                break
        else:
            print(f"grasp {name}: no room for the tips ({verdict.reason}), not tried", flush=True)
            return False
        ok = self._run(f"grasp {name}", [("up", dict(z=1.0)), ("open", dict(grip=self.open)), ("turn", dict(rot=rot)),
                                          ("over it", self._cmd(c, rot)), ("down", dict(z=0.0)), ("close", dict(grip=self.closed))])
        if ok and name not in self.env.held:  # closed on nothing
            self.goto(f"grasp {name}: nothing between the tips, raising", z=1.0)
            return False
        if ok and not self.goto(f"grasp {name}: lift", z=1.0):
            self._give_up(f"grasp {name}")
            return False
        return ok

    def place(self, xy, tip_mm: float | None = None) -> bool:
        """Carry what is held to tips position `xy`, lower (to the glass, or to tip_mm), let go, raise."""
        rot = float(self.u._target_pos[3])
        if not self._run("place", [("carry", self._cmd(xy, rot)), ("down", dict(z=0.0 if tip_mm is None else z_for_tip(tip_mm)))]):
            return False
        if not self._let_go("place"):
            self._give_up("place")
            return False
        return self.goto("place: up", z=1.0)

    def pick_place(self, name: str, xy) -> bool:
        return self.grasp(name) and self.place(xy)

    def cube_into_slot(self, cube: str) -> bool:
        """Grasp the cube and let it go just above the slot's notch, tips along the notch (the slot's short side).

        The slot stands with its notch up (16 mm tall): the cube is dropped from SLOT_DROP_MM above its top, which
        is as low as the filter lets the tips go over the slot."""
        slot = self._on_glass("slot")
        if slot is None or not self.grasp(cube):
            return False
        rot = ((_rect_angle(slot.footprint_xy)[0] + np.pi / 2) % np.pi) / np.pi
        return self._run(f"{cube} into slot", [("turn", dict(rot=rot)), ("over the slot", self._cmd(slot.xy, rot)),
                                               ("down", dict(z=z_for_tip(slot.height_mm + SLOT_DROP_MM)))]) \
            and self._let_go(f"{cube} into slot") and self.goto(f"{cube} into slot: up", z=1.0)

    def _rope_ends(self, inset_mm: float = 12.0):
        """[(grasp point, rope direction there (rad))] for the rope's two ends, from its outline in robot x, y.

        ponytail: ends = the two outline points farthest apart, which is wrong for a rope bent into a U or a loop."""
        fp = np.asarray(self.env.states["rope"].footprint_xy)
        dist = np.linalg.norm(fp[:, None] - fp[None], axis=2)
        out = []
        for e in np.unravel_index(dist.argmax(), dist.shape):
            near = fp[(dist[e] > inset_mm / 2 * cf.MM) & (dist[e] < 2 * inset_mm * cf.MM)]  # outline on both sides, a little inwards
            p = near.mean(0) if len(near) else fp[e]
            v = fp[e] - p
            out.append((p, float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 1e-6 else 0.0))
        return out

    def rope_drag(self, end: int, delta_xy) -> bool:
        """Grasp the rope near end 0 or 1 (tips across the rope), drag it by delta_xy at the glass, let go."""
        if self._on_glass("rope") is None:
            return False
        p, along = self._rope_ends()[end]
        rot = ((along + np.pi / 2) % np.pi) / np.pi
        ok = self._run("rope drag", [("up", dict(z=1.0)), ("open", dict(grip=self.open)), ("turn", dict(rot=rot)),
                                     ("over the end", self._cmd(p, rot)), ("down", dict(z=0.0)), ("close", dict(grip=self.closed))])
        if ok and "rope" not in self.env.held:
            self.goto("rope drag: nothing between the tips, raising", z=1.0)
            return False
        return ok and self._run("rope drag", [("drag", self._cmd(p + np.asarray(delta_xy, float), rot))]) \
            and self._let_go("rope drag") and self.goto("rope drag: up", z=1.0)


def wait_scale(u) -> float:
    """0 on the mock robot (tests), 1 on the real one."""
    return 0.0 if type(u.robot).__name__ == "GripperRobotMock" else 1.0
