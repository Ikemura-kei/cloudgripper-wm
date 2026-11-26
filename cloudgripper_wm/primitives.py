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
PAN_HANDLE_MM = (15.5, 13.0, 11.0)  # centre of the pan's round part to where the handle is held, tried in this order; Pan.stl: handle from 10 to 21 mm out
ROOM_MM, ROOM_PUSH_MM = 8.0, 25.0  # clutter: push length, and the room around the object it must leave (a tip is 5 mm)
ROPE_STEP_MM = 8.0  # rope drag: grasp points tried every this far along the rope, from the end inwards
SLOT_TOP_MM, SLOT_FLOOR_MM = 16.0, 11.0  # Slot.stl: prongs' top, notch floor
SLOT_PUSH_ABOVE_MM = 2.5   # cube out of slot by pushing: tips this far above the prongs (the cube's top 5 mm sticks out)
SWING_GRIP_MM, SWINGS = 9.75, 2  # cube out of slot by swinging: grip on one prong (its centre from the slot's), turns
FREE_MM = 22.0             # reset: a cube is put down at least this far from every other object's outline


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
    u /= np.linalg.norm(u)
    return (disc + lo) / 4 * cf.MM, u


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

    def _free_spot(self, name: str) -> np.ndarray:
        """A place near CENTRE at least FREE_MM from every other object's outline (the nearest to CENTRE that is)."""
        others = [np.asarray(s.footprint_xy) for n, s in self.env.states.items() if n != name and s.footprint_xy is not None]
        ring = [np.array(CENTRE) + r * np.array([np.cos(t), np.sin(t)]) for r in (0, 0.07, 0.14) for t in np.arange(0, 2 * np.pi, np.pi / 4)]
        ok = [p for p in ring if all(np.linalg.norm(o - p, axis=1).min() > FREE_MM * cf.MM for o in others)]
        return ok[0] if ok else np.array(CENTRE)

    def reset_objects(self) -> dict[str, bool]:
        """Bring every object that is far from CENTRE back towards it, farthest first: each rigid object is picked up
        and put down on a free spot (that also works against the frame, where nothing can get behind it), or pushed
        when no grasp fits (a cube in a corner); the rope is dragged from the first spot along it the tips fit.
        Returns {name: done} for what it tried.

        ponytail: a pan whose handle points at the frame and whose round part is near it, and a slot with its long side
        against the low-x or low-y frame, stay there (no grasp fits, nothing gets behind them): turn them with a
        sideways push when that happens often."""
        done = {}
        far = lambda n: float(np.linalg.norm(np.subtract(self.env.states[n].xy, CENTRE)))
        for name in sorted((n for n in self.env.states if n != "rope" and self._on_glass(n, say=False)), key=far, reverse=True):
            if far(name) <= CENTRE_RADIUS:  # read again: an earlier move may have moved it
                continue
            # carried if a grasp fits (a cube by its sides, the slot across its width, the pan by its handle), else pushed
            done[name] = self.pick_place(name, self._free_spot(name)) or self.push(
                name, np.subtract(CENTRE, self.env.states[name].xy), (far(name) - CENTRE_RADIUS) / cf.MM)
        if self._on_glass("rope", say=False):
            ends = self._rope_ends()
            i = int(np.argmax([np.linalg.norm(e - CENTRE) for e, _ in ends]))
            to = np.subtract(CENTRE, ends[i][0])
            if np.linalg.norm(to) > CENTRE_RADIUS:
                done["rope"] = self.rope_drag(i, to * (1 - CENTRE_RADIUS / np.linalg.norm(to)))
        return done

    def grasp(self, name: str, make_room: bool = True, along_mm: float = 0.0) -> bool:
        """Open, go down around `name`, close, lift. A cube is taken with the tips along one of its sides, the pan
        by its handle (8 mm wide; its round part is 20 mm and the open tips leave 22.5 mm, too little to go down around)."""
        s = self._on_glass(name)
        if s is None:
            return False
        c = np.array(s.xy, float)
        if name == "pan":
            disc, u = _pan_handle(s.footprint_xy)
            # tips either side of the handle; nearer the round part when the handle's end is at the frame
            spots = [(disc + u * k * cf.MM, np.arctan2(u[1], u[0]) + np.pi / 2) for k in PAN_HANDLE_MM]
        elif name == "slot":  # across its 15 mm width, low on the solid base (Slot.stl: the notch floor is 11 mm up)
            a = _rect_angle(s.footprint_xy)[0]  # along_mm: off the centre along its length (one end, for a swing)
            spots = [(c + along_mm * cf.MM * np.array([np.cos(a), np.sin(a)]), a + np.pi / 2)]
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
        for c, th in spots:
            rot = (th % np.pi) / np.pi
            over = [*self._cmd(c, rot).values(), 1.0, rot, self.open]
            verdict = cf.check(over, [*over[:2], 0.0, rot, self.open], self.env.states)
            if verdict.allowed:
                break
            reasons.append(verdict.reason)
        else:
            # clutter: push it or a neighbour apart once, then try again (no use when only the frame is in the way)
            if make_room and any("frame" not in r for r in reasons) and self._make_room(name):
                return self.grasp(name, make_room=False)
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

        The slot stands with its notch up (16 mm tall): the cube is dropped with the tips SLOT_DROP_MM above its top, as
        low as the filter lets a held cube go over the slot."""
        slot = self._on_glass("slot")
        if slot is None or not self.grasp(cube):
            return False
        rot = ((_rect_angle(slot.footprint_xy)[0] + np.pi / 2) % np.pi) / np.pi
        return self._run(f"{cube} into slot", [("turn", dict(rot=rot)), ("over the slot", self._cmd(slot.xy, rot)),
                                               ("down", dict(z=z_for_tip(slot.height_mm + SLOT_DROP_MM)))]) \
            and self._let_go(f"{cube} into slot") and self.goto(f"{cube} into slot: up", z=1.0)

    def cube_out_of_slot(self, cube: str, way: str = "push") -> bool:
        """Get a cube out of the slot's notch (Slot.stl: 13.5 mm wide, open at both ends along the slot's width,
        floor 11 mm up, prongs 16 mm). Four ways, to be compared on the robot (Kei, 2026-11-26):
          push   closed tips just above the prongs push the cube's top out along the notch (nothing precise)
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
            d = notch if np.dot(np.subtract(CENTRE, c), notch) >= 0 else -notch  # out of the end facing the glass centre
            rot = ((np.arctan2(d[1], d[0]) + np.pi / 2) % np.pi) / np.pi
            half = cf.FINGER_MM[1] / 2  # the tips reach this far along the push
            start, end = c - d * (7.5 + half + 3) * mm, c + d * (7.5 + 2 - half) * mm  # the cube's back face 2 mm past the slot's side
            ok = self._run(f"{cube} out of slot (push)", [("up", dict(z=1.0)), ("close", dict(grip=self.closed)), ("turn", dict(rot=rot)),
                                                            ("behind it", self._cmd(start, rot)), ("down", dict(z=z_for_tip(SLOT_TOP_MM + SLOT_PUSH_ABOVE_MM))),
                                                            ("push", self._cmd(end, rot)), ("up", dict(z=1.0))])
        elif way == "drop":
            ok = self.grasp("slot") and self.place(self._free_spot("slot"), tip_mm=cf.tip_mm(1.0))
        elif way == "swing":
            spot = self._free_spot("slot")
            if not (self.grasp("slot", along_mm=SWING_GRIP_MM) and self.goto(f"{cube} out of slot (swing): carry", **self._cmd(spot, float(self.u._target_pos[3])))):
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
        for k in np.arange(12.0, length_mm - 11, ROPE_STEP_MM):
            p, along = self._rope_ends(k)[end]
            rot = ((along + np.pi / 2) % np.pi) / np.pi
            over = [*self._cmd(p, rot).values(), 1.0, rot, self.open]
            verdict = cf.check(over, [*over[:2], 0.0, rot, self.open], self.env.states)
            if verdict.allowed:
                break
        else:
            print(f"rope drag: no room for the tips anywhere along the rope, not tried", flush=True)
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
