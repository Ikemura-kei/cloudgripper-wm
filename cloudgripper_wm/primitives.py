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
SLOT_DROP_MM = 3.0         # a cube is let go with the tips this far above the slot's top (its bottom 2.4 mm above;
                           # the filter wants CLEAR_MM between a held object's bottom and what is under it)
EDGE_SHIFT_MM = 2.0        # grasping a cube that lies against the frame: the tips go this far towards the glass
WIDE_OPEN = 1.0            # widest gripper command: 29.7 mm between tip centres, 24.7 mm between the tips (2026-11-27);
                           # only for the pan's 20 mm round part, the cell's range stays at 0.825 for everything else
PAN_HANDLE_MM = (15.5, 13.0, 11.0)  # centre of the pan's round part to where the handle is held, tried in this order; Pan.stl: handle from 10 to 21 mm out
ROOM_MM, ROOM_PUSH_MM = 8.0, 25.0  # clutter: push length, and the room around the object it must leave (a tip is 5 mm)
ROPE_STEP_MM = 8.0  # rope drag: grasp points tried every this far along the rope, from the end inwards
SLOT_TOP_MM, SLOT_FLOOR_MM = 16.0, 5.0  # Slot.stl (stored upside down): walls' top, notch floor; the notch is 13.5 mm
                           # wide and 11 mm deep, so a seated cube's top (15 mm) is below the walls
# Cube out of slot by pushing: the height command for the open tip in the notch (cube from 5 to 15 mm, floor 5 mm).
# Z_TO_TIP_MM was measured with the fingers closed; open they sit 2 to 3 mm higher (Kei, 2026-11-26). Tried: z 0 hit
# the slot's base under the notch (prim_19: tip below 5 mm), z 0.27 passed over the cube (prim_18: above 15 mm,
# where the table + 3 mm says 10.5). 0.15 is in the middle: table + 2.5 mm gives 7 mm, and if the arm stops higher
# when lowered from above, as 0.27 suggests, there are 8 mm to spare before it misses.
# ponytail: one guess between two failures; measure the open tips at 0.1 to 0.3 if 0.15 misses too
NOTCH_PUSH_Z = 0.15
NOTCH_PUSH_PAST_MM = 8.0   # the push ends with the cube's back face this far past the slot's side (2 mm left it on
                           # the edge, prim_20)
NOTCH_PUSH_TRIES = 1       # pushes while the cube is still seen in the notch afterwards. Was 3: prim_21 (2026-11-26)
                           # then carried the slot and pushed again, unattended, and a finger pressed on the slot's edge
NOTCH_PUSH_GAP_MM = 4.0    # ... and it comes down this far from the slot's side (2 mm was refused in prim_17: the tracked
                           # outline is about 1 mm larger than the slot, and the filter adds LOWER_MM)
SLOT_WALL_MM = 9.75        # centre of either 6 mm wall from the slot's centre, along its length (where it is grasped)
SWINGS = 2                 # cube out of slot by swinging: turns to and fro
RECLOSE_EASE = 0.05         # grasp: after closing, open this much and close again (a fresh command to the gripper)
LIFT_RISE_PX = 40           # grasp: a lifted object rises at least this much in the top view (the slot rose ~100 px)
DROP_PX = 50                # before lowering: a held object this far from where the carry should have put it was dropped
FREE_MM = 22.0             # reset: a cube is put down at least this far from every other object's outline
PARK = (0.5, 0.95)         # scene check: the arm looks from here at the top height, above the glass's far edge in the
                           # top view and over the frame from below: it hides nothing on the glass in either view
EDGE_MM = 10.0             # scene check: closest a rigid object may lie to the frame (the rope: some part of it farther).
                           # ponytail: first guess (a tip is 5 mm, plus the filter's margins); tune in supervised runs
PLAY_WEIGHTS = dict(pick_place=2, cube_into_slot=2, cube_out_of_slot=2, push=1, rope_drag=1)  # play: kinds (Kei, 2026-11-27)
PLAY_CARRY = ("cube_blue", "cube_white")  # play pick_place: cubes only until the slot and pan grasps have run supervised
PLAY_PUSH_MM, PLAY_DRAG_MM = (10.0, 40.0), (10.0, 30.0)  # play: push and rope-drag distances, drawn uniformly


def _clear_of(p, outlines, mm: float) -> bool:
    """Is point p (robot x, y) farther than mm from every outline (robot x, y polygons), edges included?"""
    q = tuple(float(v) for v in np.asarray(p, float) / cf.MM)
    return all(cv2.pointPolygonTest(np.float32(np.asarray(o, float) / cf.MM), q, True) < -mm for o in outlines)


def z_for_tip(mm: float) -> float:
    """The height command that puts the tips `mm` above the glass (inverse of collision_filter.tip_mm)."""
    z, t = zip(*cf.Z_TO_TIP_MM)
    return float(np.interp(mm, t, z))


def _rect_angle(footprint_xy) -> tuple[float, float]:
    """Direction (rad) of the fitted rectangle's long side in robot x, y, and its long / short ratio."""
    (_, _), (w, h), a = cv2.minAreaRect(np.float32(footprint_xy) * 1000)
    return (np.deg2rad(a) if w >= h else np.deg2rad(a + 90)) % np.pi, max(w, h) / max(min(w, h), 1e-6)


def _pan_handle(footprint_xy) -> tuple[np.ndarray, np.ndarray]:
    """Centre of the pan's round part and the handle's direction (unit vector), robot x, y. The round part's centre
    is the outline's most inward point; the outline's centre of area lies from there towards the handle."""
    fp = np.asarray(footprint_xy, float) / cf.MM * 4  # quarter millimetres
    lo = fp.min(0) - 8
    m = np.zeros(tuple(np.ceil(fp.max(0) - lo + 8).astype(int)[::-1]), np.uint8)
    cv2.fillPoly(m, [np.round(fp - lo).astype(np.int32)], 1)
    disc = np.array(cv2.minMaxLoc(cv2.GaussianBlur(cv2.distanceTransform(m, cv2.DIST_L2, 5), (0, 0), 2))[3], float)
    mo = cv2.moments(m, binaryImage=True)
    u = np.array([mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]]) - disc
    n = np.linalg.norm(u)
    return (disc + lo) / 4 * cf.MM, (u / n if n > 2 else None)  # under half a millimetre: no handle seen (mask cut short)


class Primitives:
    def __init__(self, env, on_step: Callable[[str, bool], None] | None = None) -> None:
        """`env`: a FilteredCloudGripperWrapper after reset. `on_step(label, blocked)`: called after every move."""
        self.env, self.u, self.on_step = env, env.unwrapped, on_step
        self.open, self.closed = self.u._gripper_range[1], self.u._gripper_range[0]
        self.surprise: str | None = None  # set when the cameras contradict the plan (not lifted, dropped, closed on
                                          # nothing): the collection runs a self-check next; the caller clears it

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
        """Arm to the top height, empty-handed. What is held at the bottom is let go where it is (it rests on the
        glass); what is held higher up is put back where it was picked up (that spot was free), after the drop check;
        only if the filter refuses that is it dropped from the top height. prim_28: a cube given up over the pan was
        dropped onto it from the top height."""
        if self.env.held and self.u._target_pos[2] < 0.5:
            self._let_go(f"{name}: given up")
        self.goto(f"{name}: given up, raising", z=1.0)
        back = getattr(self, "_lifted", None)
        if self.env.held and back is not None and self._still_held(f"{name}: given up") and self.env.held:
            x, y = back[1]
            if self.goto(f"{name}: given up, back where it was taken", x=float(x), y=float(y)) and \
                    self.goto(f"{name}: given up, down", z=0.0) and self._let_go(f"{name}: given up"):
                self.goto(f"{name}: given up, up", z=1.0)
                return
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

    def _on_glass(self, name: str, say: bool = True):
        s = self.env.states.get(name)
        if s is not None and s.status == "on_glass" and s.footprint_xy is not None:
            return s
        if say:
            print(f"{name} is not on the glass ({'not in the scene' if s is None else s.status + (' on ' + s.on if s.on else '')}): nothing done", flush=True)
        return None

    # ------------------------------------------------------------------ primitives
    def _push_plan(self, name: str, direction, dist_mm: float, say: bool = True):
        """(rot, tips start, tips end, tip height mm) for a push of `name`, or None when it cannot be done."""
        s = self._on_glass(name, say)
        if s is None:
            return None
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        c, fp = np.array(s.xy), np.asarray(s.footprint_xy)
        back = float(((c - fp) @ d).max())  # how far the object reaches behind its centre
        rot = ((np.arctan2(d[1], d[0]) + np.pi / 2) % np.pi) / np.pi
        x0, y0, x1, y1 = cf.GLASS
        m = cf.CLEAR_MM * cf.MM

        def inside(p):  # both closed tips over the glass, clear of the frame
            t = np.concatenate(cf.finger_polys([*self._cmd(p, rot).values(), 0, rot, self.closed]))
            return x0 + m <= t[:, 0].min() and t[:, 0].max() <= x1 - m and y0 + m <= t[:, 1].min() and t[:, 1].max() <= y1 - m

        # the run-up is shortened when the frame is close behind the object (pan 17 mm from the frame, 2026-11-26)
        starts = [c - d * (back + (cf.FINGER_MM[1] / 2 + a) * cf.MM) for a in (APPROACH_MM, 6.0, 3.0)]
        start = next((p for p in starts if inside(p)), starts[0])
        end = c + d * (dist_mm * cf.MM - back - cf.FINGER_MM[1] / 2 * cf.MM)
        tip = PUSH_TIP_MM
        if not inside(start):  # the push starts over the frame
            if s.height_mm < 10:
                if say:
                    print(f"push {name}: the frame is right behind it and it is too low to catch from over the frame: nothing done", flush=True)
                return None
            tip = OVER_FRAME_TIP_MM
        (ax, bx), (ay, by) = self.u._x_range, self.u._y_range
        if not all(ax <= q["x"] <= bx and ay <= q["y"] <= by for q in (self._cmd(start, rot), self._cmd(end, rot))):
            if say:
                print(f"push {name}: the arm cannot reach behind it (outside its range): nothing done", flush=True)
            return None
        return rot, start, end, tip

    def _push_fits(self, plan) -> bool:
        """Does the filter allow the lowering behind the object and the push itself, as the scene is now?"""
        rot, start, end, tip = plan
        over, z = [*self._cmd(start, rot).values(), 1.0, rot, self.closed], z_for_tip(tip)
        down = [*over[:2], z, rot, self.closed]
        return cf.check(over, down, self.env.states).allowed and cf.check(down, [*self._cmd(end, rot).values(), z, rot, self.closed], self.env.states).allowed

    def push(self, name: str, direction, dist_mm: float) -> bool:
        """Push `name` by dist_mm along `direction` (robot x, y) with the closed gripper held across the direction."""
        plan = self._push_plan(name, direction, dist_mm)
        if plan is None:
            return False
        rot, start, end, tip = plan
        return self._run(f"push {name}", [("up", dict(z=1.0)), ("close", dict(grip=self.closed)), ("turn", dict(rot=rot)),
                                           ("behind it", self._cmd(start, rot)), ("down", dict(z=z_for_tip(tip))),
                                           ("push", self._cmd(end, rot)), ("up", dict(z=1.0))])

    def _make_room(self, name: str) -> bool:
        """A grasp of `name` found no room for the tips: push it, or a rigid neighbour, ROOM_PUSH_MM so that it ends
        with ROOM_MM clear all round. Every push in eight directions is asked of the filter first; the one that
        leaves most room is done. False (nothing moved) when none fits.

        ponytail: the room afterwards is predicted by shifting the pushed outline only; what it pushes along is
        ignored, and the rope is never the one moved. Two objects that touch usually leave no push that fits."""
        st = {n: s for n, s in self.env.states.items() if s.status == "on_glass" and s.footprint_xy is not None}
        mm = lambda fp: np.float32(np.asarray(fp) / cf.MM)
        gap = lambda a, b: min(-max(cv2.pointPolygonTest(b, tuple(map(float, p)), True) for p in a),
                               -max(cv2.pointPolygonTest(a, tuple(map(float, p)), True) for p in b))
        x0, y0, x1, y1 = cf.GLASS
        near = [n for n in st if n not in (name, "rope") and gap(mm(st[name].footprint_xy), mm(st[n].footprint_xy)) < 20]
        best = (ROOM_MM, None)
        for mover in [name, *near]:
            for t in np.arange(0, 2 * np.pi, np.pi / 4):
                d = np.array([np.cos(t), np.sin(t)])
                fps = {n: np.asarray(s.footprint_xy) + (d * ROOM_PUSH_MM * cf.MM if n == mover else 0) for n, s in st.items()}
                f = fps[mover]
                if f[:, 0].min() < x0 + 0.07 or f[:, 0].max() > x1 - 0.07 or f[:, 1].min() < y0 + 0.07 or f[:, 1].max() > y1 - 0.07:
                    continue  # would end within 10 mm of the frame
                room = min(gap(mm(fps[name]), mm(fps[n])) for n in fps if n != name)
                if room > best[0]:
                    plan = self._push_plan(mover, d, ROOM_PUSH_MM, say=False)
                    if plan is not None and self._push_fits(plan):
                        best = (room, (mover, d))
        if best[1] is None:
            return False
        print(f"grasp {name}: making room by pushing {best[1][0]}", flush=True)
        return self.push(best[1][0], best[1][1], ROOM_PUSH_MM)

    def _free_spot(self, name: str) -> np.ndarray | None:
        """The place nearest CENTRE, EDGE_MM + 10 mm from the frame, that is at least FREE_MM from every other object's
        outline (edges included; prim_28: corners only put a cube on the pan's edge), or None: no free place."""
        others = [np.asarray(s.footprint_xy) for n, s in self.env.states.items() if n != name and getattr(s, "footprint_xy", None) is not None]
        x0, y0, x1, y1 = cf.GLASS
        m = (EDGE_MM + 10) * cf.MM
        ring = [np.array(CENTRE) + r * np.array([np.cos(t), np.sin(t)]) for r in (0, 0.07, 0.14, 0.21, 0.28) for t in np.arange(0, 2 * np.pi, np.pi / 8)]
        ok = [p for p in ring if x0 + m < p[0] < x1 - m and y0 + m < p[1] < y1 - m and _clear_of(p, others, FREE_MM)]
        return ok[0] if ok else None

    def reset_objects(self) -> list[str]:
        """Make the scene fit for play with as little moving as possible (Kei, 2026-11-27): only the objects the scene
        check names, one at a time, the scene checked again after each. Returns the problems left (none: fit).

        A rigid object is moved just clear of the frame (EDGE_MM + 10 mm from it, towards the centre): slid (clamped
        where it lies; prim_28: that freed the pan from the frame), else pushed, else carried to a free spot, but never
        the pan (carried by its handle it tilts: the round part is the heavy end). The slot is pushed first (Kei: it
        can nearly always be pushed). The rope is dragged in. An object
        that is not on the glass cannot be brought back here: it stays a problem, for a person.

        prim_28, before this: every object more than 21 mm from the centre was carried there; that tilted the pan,
        pushed the slot and rope to the frame and put the white cube on the pan."""
        problems, tried = self.scene_check(), set()
        while problems:
            name = next((n for n in (p.split(":")[0] for p in problems) if n not in tried), None)
            if name is None:
                break
            tried.add(name)
            s = self._on_glass(name, say=False)
            if s is None:
                print(f"reset: {name} is not on the glass, nothing to do here", flush=True)
                continue
            d = np.subtract(CENTRE, s.xy)
            d = d / max(np.linalg.norm(d), 1e-9)
            mm = max(10.0, EDGE_MM + 10 - self._edge_mm(name, s))
            print(f"reset: {name}, {self._edge_mm(name, s):.0f} mm from the frame: moving it {mm:.0f} mm in", flush=True)
            if name == "rope":
                self.rope_drag(0, tuple(d * mm * cf.MM))
            else:
                ways = [lambda: self.slide(name, d, mm), lambda: self.push(name, d, mm)]
                if name == "slot":
                    ways.reverse()
                if name != "pan":
                    ways.append(lambda: self.pick_place(name, self._free_spot(name)))
                for way in ways:  # each with the object still on the glass (one given up may have moved it)
                    if self._on_glass(name, say=False) is None or way():
                        break
            problems = self.scene_check()
        return problems

    def _edge_mm(self, name: str, s) -> float:
        """How far (mm) `s` lies from the frame: its nearest point; for the rope its farthest point (scene check)."""
        x0, y0, x1, y1 = cf.GLASS
        p = np.atleast_2d(s.footprint_xy)
        d = np.min([p[:, 0] - x0, x1 - p[:, 0], p[:, 1] - y0, y1 - p[:, 1]], axis=0) / cf.MM
        return float(d.max() if name == "rope" else d.min())

    def scene_check(self) -> list[str]:
        """Is the scene fit for a play episode (Kei, 2026-11-27)? Parks the arm at PARK, looks again, and returns the
        problems, none when fit: every object accounted for, on the glass or a cube in the slot's notch; no rigid
        object within EDGE_MM of the frame; the rope only when all of it is (Kei: it is dragged from wherever along it
        the tips fit, so a rope with any part well inside is fine). On problems the caller runs reset_objects (it fixes
        what is named here, checking again after each), then stops for a person if problems are left."""
        from cloudgripper_wm.perception.scene_tracker import INVENTORY  # here: that module loads torch
        if not (self.goto("scene check: up", z=1.0) and self.goto("scene check: park", x=PARK[0], y=PARK[1])):
            return ["could not park: the filter blocked the move"]
        states, problems = self.env.look_again(), []
        for name in INVENTORY:
            s = states.get(name)
            if s is None or not (s.status == "on_glass" or (s.status == "stacked" and s.on == "slot" and getattr(s, "in_notch", False))):
                problems.append(f"{name}: {'not tracked' if s is None else s.status + (' on ' + s.on if s.on else '')}")
            elif s.status == "on_glass":
                edge = self._edge_mm(name, s)
                if edge < EDGE_MM:
                    problems.append(f"{name}: {edge:.0f} mm from the frame (needs {EDGE_MM:.0f})")
        return problems

    def _random_spot(self, name: str, rng) -> np.ndarray | None:
        """A random place on the glass for `name`, EDGE_MM + 10 mm from the frame and FREE_MM from every other object's
        outline, or None when 50 draws find none."""
        others = [np.asarray(s.footprint_xy) for n, s in self.env.states.items() if n != name and getattr(s, "footprint_xy", None) is not None]
        x0, y0, x1, y1 = cf.GLASS
        m = (EDGE_MM + 10) * cf.MM
        for _ in range(50):
            p = rng.uniform((x0 + m, y0 + m), (x1 - m, y1 - m))
            if _clear_of(p, others, FREE_MM):
                return p
        return None

    def random_primitive(self, rng) -> tuple[str, tuple] | None:
        """A primitive and random arguments that the scene allows now, or None: the kind by PLAY_WEIGHTS among the
        kinds that have a candidate, then one of its candidates. Targets keep objects EDGE_MM + 5 mm from the frame,
        so that play does not leave something where no primitive can recover it (the pan by the frame, prim_26)."""
        st = self.env.states
        on = lambda n: n in st and st[n].status == "on_glass" and st[n].footprint_xy is not None
        cubes = ("cube_blue", "cube_white")
        notch = [n for n in cubes if n in st and st[n].status == "stacked" and st[n].on == "slot" and getattr(st[n], "in_notch", False)]
        x0, y0, x1, y1 = cf.GLASS
        m = (EDGE_MM + 5) * cf.MM
        inside = lambda fp: fp[:, 0].min() > x0 + m and fp[:, 0].max() < x1 - m and fp[:, 1].min() > y0 + m and fp[:, 1].max() < y1 - m
        unit = lambda t: np.array([np.cos(t), np.sin(t)])
        cand = {k: [] for k in PLAY_WEIGHTS}
        for n in PLAY_CARRY:
            spot = self._random_spot(n, rng) if on(n) else None
            if spot is not None:
                cand["pick_place"].append((n, tuple(float(v) for v in spot)))
        if on("slot") and not notch:
            cand["cube_into_slot"] += [(n,) for n in cubes if on(n)]
        cand["cube_out_of_slot"] += [(n, "push") for n in notch]
        for n in ("cube_blue", "cube_white", "slot", "pan"):
            d, dist = unit(rng.uniform(0, 2 * np.pi)), float(rng.uniform(*PLAY_PUSH_MM))
            if on(n) and inside(np.asarray(st[n].footprint_xy) + d * dist * cf.MM):
                cand["push"].append((n, tuple(float(v) for v in d), dist))
        if on("rope"):
            cand["rope_drag"].append((int(rng.integers(2)), tuple(float(v) for v in unit(rng.uniform(0, 2 * np.pi)) * rng.uniform(*PLAY_DRAG_MM) * cf.MM)))
        kinds = [k for k in PLAY_WEIGHTS if cand[k]]
        if not kinds:
            return None
        w = np.array([PLAY_WEIGHTS[k] for k in kinds], float)
        kind = kinds[rng.choice(len(kinds), p=w / w.sum())]
        return kind, cand[kind][rng.integers(len(cand[kind]))]

    def grasp(self, name: str, make_room: bool = True, along_mm: float | None = None, lift: bool = True) -> bool:
        """Open, go down around `name`, close, lift. A cube is taken with the tips along one of its sides, the pan
        by its handle (8 mm wide). `lift=False`: stop once it is clamped, at the bottom (`slide`); the pan may then
        also be clamped by its round part (20 mm) with the gripper opened to WIDE_OPEN (24.7 mm between the tips; 22.5
        at the usual open is too little) when no handle spot fits, as with the handle at the frame (Kei, 2026-11-27;
        prim_26). It does not lift that way (prim_27: rose 5 px), so a lifting grasp never takes it."""
        self._lifted = None
        s = self._on_glass(name)
        if s is None:
            return False
        c = np.array(s.xy, float)
        if name == "pan":
            disc, u = _pan_handle(s.footprint_xy)
            # tips either side of the handle; nearer the round part when the handle's end is at the frame
            # (no handle seen: round part only, every direction)
            a = 0.0 if u is None else np.arctan2(u[1], u[0])
            spots = [] if u is None else [(disc + u * k * cf.MM, a + np.pi / 2) for k in PAN_HANDLE_MM]
            # round part, for a clamp only (it does not lift, prim_27): a fan of directions off the handle (along it,
            # a tip lands on the handle); the filter picks
            fan = [1 / 2, 3 / 8, -3 / 8, 1 / 4, -1 / 4, 1 / 8, -1 / 8] + ([] if u is not None else [0])
            spots += [] if lift else [(disc, a + t, WIDE_OPEN) for t in np.pi * np.array(fan)]
        elif name == "slot":  # across its 15 mm width, on one of its walls (the middle is only the 5 mm base under the notch)
            a = _rect_angle(s.footprint_xy)[0]
            spots = [(c + k * cf.MM * np.array([np.cos(a), np.sin(a)]), a + np.pi / 2)
                     for k in ([along_mm] if along_mm is not None else [SLOT_WALL_MM, -SLOT_WALL_MM])]
        else:
            # Near the frame the tips go along it: the open tips reach 16 mm either way and would be over the frame
            # otherwise. Right against it they also go EDGE_SHIFT_MM towards the glass (the wide side of a tip would
            # be over the frame): they still hold 8 of the cube's 10 mm. In a corner neither direction fits.
            th = _rect_angle(s.footprint_xy)[0] % (np.pi / 2)
            x0, y0, x1, y1 = cf.GLASS
            reach, near = 19 * cf.MM, 7 * cf.MM
            at_x, at_y = c[0] < x0 + reach or c[0] > x1 - reach, c[1] < y0 + reach or c[1] > y1 - reach
            ths = [th, th + np.pi / 2]  # the cube's two side directions
            if at_x != at_y:  # one frame side: the direction along that side first
                ths.sort(key=lambda t: abs(np.sin(t)) if at_x else abs(np.cos(t)), reverse=True)
            c += EDGE_SHIFT_MM * cf.MM * np.array([int(c[0] < x0 + near) - int(c[0] > x1 - near), int(c[1] < y0 + near) - int(c[1] > y1 - near)])
            spots = [(c, t) for t in ths]
        # Ask the filter before moving: take the first spot and direction in which the open tips can go down around the cube
        # (another may put a tip on the rope or a neighbour). If none can, give up without moving.
        reasons = []
        for c, th, *wide in spots:
            rot, grip = (th % np.pi) / np.pi, wide[0] if wide else self.open
            over = [*self._cmd(c, rot).values(), 1.0, rot, grip]
            verdict = cf.check(over, [*over[:2], 0.0, rot, grip], self.env.states)
            if verdict.allowed:
                break
            reasons.append(verdict.reason)
        else:
            # clutter: push it or a neighbour apart once, then try again (no use when only the frame is in the way)
            if make_room and any("frame" not in r for r in reasons) and self._make_room(name):
                return self.grasp(name, make_room=False, lift=lift)
            print(f"grasp {name}: no room for the tips ({verdict.reason}), not tried", flush=True)
            return False
        # close, then ease off and close again (Kei, 2026-11-26): a gripper stopped by a wide object gets a fresh command
        rng0 = self.u._gripper_range
        self.u._gripper_range = (rng0[0], max(rng0[1], grip))  # the round-part grasp only: the env clips to the range
        try:
            ok = self._run(f"grasp {name}", [("up", dict(z=1.0)), ("open", dict(grip=grip)), ("turn", dict(rot=rot)),
                                              ("over it", self._cmd(c, rot)), ("down", dict(z=0.0)), ("close", dict(grip=self.closed)),
                                              ("ease", dict(grip=self.closed + RECLOSE_EASE)), ("close again", dict(grip=self.closed))])
        finally:
            self.u._gripper_range = rng0
        if ok and name not in self.env.held:  # closed on nothing
            self.surprise = f"grasp {name}: closed on nothing"
            self.goto(f"grasp {name}: nothing between the tips, raising", z=1.0)
            return False
        if not lift:
            return ok
        before = self._top_c(name)
        if ok and not self.goto(f"grasp {name}: lift", z=1.0):
            self._give_up(f"grasp {name}")
            return False
        # did it come up? In the top view a lifted object rises (about 100 px for the slot, prim_21). prim_21: the slot
        # slipped out on the lift, "holding slot" stayed, and the arm was later lowered onto it.
        after = self._top_c(name)
        if ok and before is not None and after is not None:
            print(f"grasp {name}: rose {before[1] - after[1]:.0f} px in the top view (needs {LIFT_RISE_PX})", flush=True)
        if ok and before is not None and after is not None and before[1] - after[1] < LIFT_RISE_PX:
            print(f"grasp {name}: it did not come up (top view: {before[1] - after[1]:.0f} px), letting go", flush=True)
            self.surprise = f"grasp {name}: did not come up"
            self.goto(f"grasp {name}: not lifted, opening", grip=self.open)  # opening also clears "holding"
            return False
        self._lifted = (after, np.array(self.u._target_pos[:2], float)) if ok else None
        return ok

    def slide(self, name: str, direction, dist_mm: float) -> bool:
        """Clamp `name` where it lies (the grasp's spots, and its wide opening for the pan's round part), slide it
        dist_mm along `direction` on the glass, let go and go up (Kei, 2026-11-27). The pan's round part does not
        lift (prim_27: it rose 5 px), but it can be slid away from the frame, after which a grasp or a push has room;
        a loose clamp still moves it, the tip behind it pushing."""
        if not self.grasp(name, lift=False):
            return False
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        x, y = np.asarray(self.u._target_pos[:2], float) + d * dist_mm * cf.MM
        return self._run(f"slide {name}", [("slide", dict(x=float(x), y=float(y))), ("let go", dict(grip=self.open)), ("up", dict(z=1.0))])

    def _top_c(self, name: str):
        """Centre of `name`'s top-view mask (x, y px), or None."""
        m = getattr(self.env.states.get(name), "mask_top", None)
        return None if m is None or not m.any() else np.argwhere(m)[:, ::-1].mean(0)

    def _still_held(self, label: str) -> bool:
        """Before lowering what is held: is it still where a held object should be after the carry? A drop shows
        as the object far below that place in the top view. If it was dropped, "holding" is cleared (the filter
        then sees it as an obstacle again) and False is returned. Unknown (no top-view mask) counts as held."""
        tr, held = self.env._tracker, list(self.env.held)
        if not held or getattr(self, "_lifted", None) is None or not hasattr(tr, "top_px"):
            return True
        c0, xy0 = self._lifted
        c1 = self._top_c(held[0])
        if c0 is None or c1 is None:
            return True
        expect = c0 + tr.top_px(self.u._target_pos[:2]) - tr.top_px(xy0)
        print(f"{label}: {held[0]} is {np.round(c1 - expect)} px from where the carry should have put it (allowed {DROP_PX})", flush=True)
        if abs(c1[1] - expect[1]) <= DROP_PX and abs(c1[0] - expect[0]) <= DROP_PX:
            return True
        print(f"{label}: {held[0]} is not where it should be in the gripper (top view, {c1 - expect} px off): dropped", flush=True)
        self.env.held = []
        self.surprise = f"{label}: {held[0]} dropped"
        return False

    def place(self, xy, tip_mm: float | None = None) -> bool:
        """Carry what is held to tips position `xy`, lower (to the glass, or to tip_mm), let go, raise. xy None (no
        free place): given up, what is held goes back where it was taken."""
        if xy is None:
            print("place: no free place", flush=True)
            self._give_up("place")
            return False
        rot = float(self.u._target_pos[3])
        if not self._run("place", [("carry", self._cmd(xy, rot))]):
            return False
        if not self._still_held("place"):
            return False
        if not self._run("place", [("down", dict(z=0.0 if tip_mm is None else z_for_tip(tip_mm)))]):
            return False
        if not self._let_go("place"):
            self._give_up("place")
            return False
        return self.goto("place: up", z=1.0)

    def pick_place(self, name: str, xy) -> bool:
        if xy is None:
            print(f"pick_place {name}: no free place to put it, not tried", flush=True)
            return False
        return self.grasp(name) and self.place(xy)

    def cube_into_slot(self, cube: str) -> bool:
        """Grasp the cube and let it go just above the slot's notch, tips along the notch (the slot's short side).

        The slot stands with its notch up (16 mm tall): the cube is dropped with the tips SLOT_DROP_MM above its top, as
        low as the filter lets a held cube go over the slot."""
        slot = self._on_glass("slot")
        if slot is None or not self.grasp(cube):
            return False
        rot = ((_rect_angle(slot.footprint_xy)[0] + np.pi / 2) % np.pi) / np.pi
        return self._run(f"{cube} into slot", [("turn", dict(rot=rot)), ("over the slot", self._cmd(slot.xy, rot))]) \
            and self._still_held(f"{cube} into slot") \
            and self._run(f"{cube} into slot", [("down", dict(z=z_for_tip(slot.height_mm + SLOT_DROP_MM)))]) \
            and self._let_go(f"{cube} into slot") and self.goto(f"{cube} into slot: up", z=1.0)

    def _notch_push(self, cube: str):
        """(start pose, end pose, rot, z) for pushing `cube` out of the slot's notch with one open tip, or why not."""
        slot, cs = self._on_glass("slot", say=False), self.env.states.get(cube)
        if slot is None or cs is None or not (cs.status == "stacked" and cs.on == "slot" and getattr(cs, "in_notch", False)):
            return "it is not seen in the slot's notch"
        c, a = np.array(slot.xy, float), _rect_angle(slot.footprint_xy)[0]
        along, notch, mm = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)]), cf.MM
        d = notch if np.dot(np.subtract(CENTRE, c), notch) >= 0 else -notch  # out of the end facing the glass centre
        rot = (a % np.pi) / np.pi  # tip axis along the slot's length: each tip is 5 mm across the notch
        half = cf.FINGER_MM[1] / 2  # a tip reaches this far along the push
        z = NOTCH_PUSH_Z
        (ax, bx), (ay, by) = self.u._x_range, self.u._y_range
        reason = ""
        for side in (1, -1):  # the other tip beside one end of the slot or the other: the first the filter allows
            mid = along * side * cf.sep_mm(self.open) / 2 * mm  # tips' midpoint from the tip in the notch
            start, end = c - d * (7.5 + half + NOTCH_PUSH_GAP_MM) * mm + mid, c + d * (7.5 + NOTCH_PUSH_PAST_MM - half) * mm + mid  # cube's back face this far out
            p0 = [*self._cmd(start, rot).values(), 1.0, rot, self.open]
            p1, p2 = [*p0[:2], z, rot, self.open], [*self._cmd(end, rot).values(), z, rot, self.open]
            v = cf.check(p0, p1, self.env.states)
            v = cf.check(p1, p2, self.env.states) if v.allowed else v
            if v.allowed and all(ax <= q[0] <= bx and ay <= q[1] <= by for q in (p0, p2)):
                return p0, p2, rot, z
            reason = v.reason if not v.allowed else "outside the arm's range"
        return f"no room for the tips ({reason})"

    def cube_out_of_slot(self, cube: str, way: str = "push", tries: int = NOTCH_PUSH_TRIES, move_slot: bool = False) -> bool:
        """Get a cube out of the slot's notch (Slot.stl: 13.5 mm wide, open at both ends along the slot's width,
        floor 5 mm up, walls 16 mm; a seated cube's top is 1 mm below them). Four ways, to be compared (Kei, 2026-11-26):
          push   one open tip, its 5 mm side across the notch (4.25 mm to spare each side), goes along the notch
                 2.5 mm above its floor and pushes the cube out; the other tip runs beside the slot's end
          drop   grasp the slot, carry it to a free spot, let it fall from the top: the cube bounces out
          swing  grasp the slot at one end, turn the gripper to and fro at the top height: the cube slides out
          grasp  open tips beside the notch's open ends, closed into the notch on the cube (+-1.5 mm across it)"""
        slot, cs = self._on_glass("slot"), self.env.states.get(cube)
        if slot is None or cs is None or not (cs.status == "stacked" and cs.on == "slot" and getattr(cs, "in_notch", False)):
            print(f"{cube} out of slot: it is not seen in the slot's notch: nothing done", flush=True)
            return False
        c, a = np.array(slot.xy, float), _rect_angle(slot.footprint_xy)[0]
        along, notch = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        mm = cf.MM
        if way == "push":
            plan = self._notch_push(cube)
            if isinstance(plan, str) and not move_slot:
                print(f"{cube} out of slot (push): {plan}, not tried (move_slot=True carries the slot out first)", flush=True)
                return False
            if isinstance(plan, str):  # no room where the slot is (its notch towards the frame): carry it out first
                print(f"{cube} out of slot (push): {plan}; moving the slot first", flush=True)
                if not (self.pick_place("slot", self._free_spot("slot")) and self.goto(f"{cube} out of slot: home to look", x=0.5, y=0.5)):
                    return False
                self.env.look_again()
                plan = self._notch_push(cube)
            if isinstance(plan, str):
                print(f"{cube} out of slot (push): {plan}, not tried", flush=True)
                return False
            p0, p2, rot, z = plan
            ok = self._run(f"{cube} out of slot (push)", [("up", dict(z=1.0)), ("open", dict(grip=self.open)), ("turn", dict(rot=rot)),
                                                            ("behind it", dict(x=p0[0], y=p0[1])), ("down", dict(z=z)),
                                                            ("push", dict(x=p2[0], y=p2[1])), ("up", dict(z=1.0))])
        elif way == "drop":
            ok = self.grasp("slot") and self.place(self._free_spot("slot"), tip_mm=cf.tip_mm(1.0))
        elif way == "swing":
            spot = self._free_spot("slot")
            if spot is None or not (self.grasp("slot", along_mm=SLOT_WALL_MM) and self.goto(f"{cube} out of slot (swing): carry", **self._cmd(spot, float(self.u._target_pos[3])))):
                return False
            r0 = float(self.u._target_pos[3])
            for k in range(SWINGS):  # 90 deg each way, in steps of the play step limit
                r1 = r0 + 0.5 if r0 + 0.5 <= 1.0 else r0 - 0.5
                if not (self.goto(f"{cube} out of slot (swing) {k + 1}", rot=r1) and self.goto(f"{cube} out of slot (swing) {k + 1} back", rot=r0)):
                    self._give_up(f"{cube} out of slot (swing)")
                    return False
            ok = self.place(spot)  # back at the carry's rotation: the tips are over the spot again
        elif way == "grasp":
            rot = ((np.arctan2(notch[1], notch[0])) % np.pi) / np.pi  # tips at the notch's two open ends
            ok = self._run(f"{cube} out of slot (grasp)", [("up", dict(z=1.0)), ("open", dict(grip=self.open)), ("turn", dict(rot=rot)),
                                                           ("over it", self._cmd(c, rot)), ("down", dict(z=z_for_tip(SLOT_FLOOR_MM + 2.5))),
                                                           ("close", dict(grip=self.closed)), ("lift", dict(z=1.0))])
            # ponytail: the filter takes the slot, not the cube, as clamped (the cube has no outline of its own up
            # there); harmless here, the slot is only "held" until the cube is let go
            ok = ok and self.place(self._free_spot(cube))
        else:
            raise ValueError(f"way must be push, drop, swing or grasp, not {way!r}")
        # did it come out? From home, where nothing hides the slot (the tracker only knows after looking again)
        if not (ok and self.goto(f"{cube} out of slot: back home to look", x=0.5, y=0.5)):
            return False
        st = self.env.look_again()[cube]
        if st.status != "on_glass":
            print(f"{cube} out of slot ({way}): still {st.status}{' on ' + st.on if st.on else ''} afterwards", flush=True)
            if way == "push" and tries > 1 and getattr(st, "in_notch", False):  # moved but not out: push again
                return self.cube_out_of_slot(cube, way, tries - 1, move_slot)
        return st.status == "on_glass"

    def _rope_ends(self, inset_mm: float = 12.0):
        """[(point on the rope inset_mm in from the end, rope direction there (rad))] for the rope's two ends, from its
        outline in robot x, y (resampled every 1 mm: approxPolyDP leaves long straight runs without points).

        ponytail: ends = the two outline points farthest apart, distances straight-line: wrong for a rope bent into
        a U or a loop."""
        fp0 = np.asarray(self.env.states["rope"].footprint_xy, float)
        loop = np.vstack([fp0, fp0[:1]])
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))]
        t = np.arange(0, arc[-1], cf.MM)
        fp = np.c_[np.interp(t, arc, loop[:, 0]), np.interp(t, arc, loop[:, 1])]
        dist = np.linalg.norm(fp[:, None] - fp[None], axis=2)
        out = []
        for e in np.unravel_index(dist.argmax(), dist.shape):
            band = lambda lo, hi: fp[(dist[e] > lo * cf.MM) & (dist[e] < hi * cf.MM)]  # outline on both sides
            near, back, fwd = band(inset_mm - 4, inset_mm + 4), band(max(inset_mm - 12, 0), inset_mm), band(inset_mm, inset_mm + 12)
            p = near.mean(0) if len(near) else fp[e]
            v = back.mean(0) - fwd.mean(0) if len(back) and len(fwd) else fp[e] - p  # towards the end
            out.append((p, float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 1e-6 else 0.0))
        return out

    def rope_drag(self, end: int, delta_xy) -> bool:
        """Grasp the rope near end 0 or 1 (tips across the rope), drag it by delta_xy at the glass, let go. The grasp
        point is the first one, from 12 mm in and then every ROPE_STEP_MM along the whole rope, where the filter lets
        the open tips go down (a rope lying along the frame is taken where it leaves it); none: nothing is done."""
        if self._on_glass("rope") is None:
            return False
        fp = np.asarray(self.env.states["rope"].footprint_xy)
        length_mm = np.linalg.norm(fp[:, None] - fp[None], axis=2).max() / cf.MM
        reasons = []
        for k in np.arange(12.0, length_mm - 11, ROPE_STEP_MM):
            p, along = self._rope_ends(k)[end]
            rot = ((along + np.pi / 2) % np.pi) / np.pi
            over = [*self._cmd(p, rot).values(), 1.0, rot, self.open]
            verdict = cf.check(over, [*over[:2], 0.0, rot, self.open], self.env.states)
            if verdict.allowed:
                break
            reasons.append(verdict.reason)
        else:
            why = "; ".join(f"{r} ({reasons.count(r)}x)" for r in dict.fromkeys(reasons))
            print(f"rope drag: no room for the tips anywhere along the rope ({length_mm:.0f} mm, {len(reasons)} spots: {why}), not tried", flush=True)
            return False
        ok = self._run("rope drag", [("up", dict(z=1.0)), ("open", dict(grip=self.open)), ("turn", dict(rot=rot)),
                                     (f"over the rope {k:.0f} mm from the end", self._cmd(p, rot)), ("down", dict(z=0.0)), ("close", dict(grip=self.closed))])
        if ok and "rope" not in self.env.held:
            self.goto("rope drag: nothing between the tips, raising", z=1.0)
            return False
        return ok and self._run("rope drag", [("drag", self._cmd(p + np.asarray(delta_xy, float), rot))]) \
            and self._let_go("rope drag") and self.goto("rope drag: up", z=1.0)


def wait_scale(u) -> float:
    """0 on the mock robot (tests), 1 on the real one."""
    return 0.0 if type(u.robot).__name__ == "GripperRobotMock" else 1.0
