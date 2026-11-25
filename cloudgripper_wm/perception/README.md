# Perception for cells 20 and 23 and their five objects

`SceneTracker` finds, names and tracks the objects of **one specific setup** with SAM 2.1 in both camera views.
It is written around that setup and does **not** generalise: more, fewer or other objects, another rope, another
cell or other lighting each need changes here. Every such assumption is listed below, with where it lives in
`scene_tracker.py`.

```python
from cloudgripper_wm.perception import SceneTracker

tracker = SceneTracker("robot20")            # the development cell since 2026-11-25; "robot23" also works
states = tracker.start(base_bgr, top_bgr)     # at a parking pose: find and name the objects, about 0.8 s
states = tracker.step(base_bgr)               # every control step, base view only, about 30 ms
states = tracker.step(base_bgr, top_bgr)      # both views, about 63 ms
states = tracker.refresh(base_bgr, top_bgr)   # at a parking pose: look again for objects not on the glass, 0.15 to 0.7 s
```

`base_bgr` is the undistorted base image (640x480), `top_bgr` the top image (1280x720). `states[name]` is an
`ObjectState`: status, robot x, y, in-plane angle, footprint polygon in robot coordinates, height, and both masks.

Needs a CUDA GPU and the `perception` extra (`sam2`). Timings are on an RTX 4090 with five objects; the weights
download from Hugging Face on first use (small 180 MB, large 900 MB).

## The setup it assumes

| Object | Inventory name | Size (mm) | Colour | Reference model |
|---|---|---|---|---|
| Cube | `cube_blue` | 10 x 10 x 10 | navy blue | `assets/Cube_10_10_10.stl` |
| Cube | `cube_white` | 10 x 10 x 10 | white | `assets/Cube_10_10_10.stl` |
| Slot block | `slot` | 25.5 x 15 x 16 | red | `assets/Slot.stl` |
| Pan | `pan` | 20 x 31 x 7 | green | `assets/Pan.stl` |
| Rope | `rope` | about 8 thick | lilac | none |

Cells 20 and 23: blue frame around the glass, yellow gripper, blue and grey robot parts, orange cable. Cell 20 is
the development cell since 2026-11-25 (cell 23's gripper is off its servo, cell 22's height axis does not respond).

## Cell 20: colour prompts first

On cell 20 (`self.colour_prompts`) every object is prompted by its colour, in each view on its own:

1. **Colour region** (`colour_region`, `COLOUR`, `AREA_PX`). Inside the glass area, the pixels of the object's colour
   must form exactly one connected region of about the object's size (cubes also compact). No region, or more than
   one, gives no colour prompt: a doubtful region never becomes a prompt.
2. **Points** (`spread_points`). Three points well inside the region (five along the rope, from its largest colour
   patch) are the SAM 2 prompt.
3. **Post-filter** (`_update`, `one_piece`). Every mask is cut to its largest connected region and dropped if that
   is larger than the object can be; rigid masks first lose thin tails (7x7 opening), such as a wire of the same
   colour behind the object.
4. **Fallback.** What colour did not find goes through the shape steps below (promptless pass, homography boxes).

A cube's colour region and its mask in the base view must also be one solid square of a cube's size (`cube_like`):
on a second layout (2026-11-25) the white cube lay against the gantry rail, its own colour region failed, and a grey
rail mount of about the right area was taken for it. With the check the mount is rejected, the shape fallback finds
the cube, and a cube mask that later runs onto a robot part counts as not seen from below.

Known limits: the HSV numbers come from two layouts and one lighting; from below the white cube has the gantry
rail's grey and the blue cube the frame's and the cable chain's blue, so a cube touching those in the picture is not
found by colour (the shape fallback has to find it); the rope keeps only its largest piece (no merging across an occlusion yet).

Cell 23 still runs the shape steps only.

## How it works (shape steps: cell 23, and the fallback on cell 20)

1. **Promptless pass, base view** (`_look` step 1). SAM 2's automatic mask generator proposes masks on the glass.
   `assign_inventory` names them by the fitted rectangle's sides and the solidity against the templates in `RIGID`.
2. **Rope** (`_rope_points`). A colour seed gives five points along the rope; SAM 2 turns them into the mask.
3. **Top view from the base view** (`_look`, `CELLS[...]["base_to_top"]`). Each base mask is projected into the top
   view by a homography and becomes a box prompt there. The top view has no detector of its own.
4. **Hidden cubes** (`_look` step 3). A cube not found on the glass is searched in the top view as a compact patch
   of another colour inside a located object's mask (cube on the pan, cube in the slot).
5. **Tracking** (`_Stream`). SAM 2's video predictor is fed one frame at a time; memory use stays flat.
6. **Status** (`_update`), from the known inventory:

| Base view | Top view | Status |
|---|---|---|
| mask present | any | `on_glass` |
| no mask | mask touching another object's mask | `stacked`, with `on` naming that object |
| no mask | mask touching nothing | `off_glass` (for example on the frame) |
| no mask | no mask | `unaccounted`: stop and look, do not continue blind |

A cube that is `stacked` or `off_glass` also gets `inferred_px`: where it would be in the base view, from its
offset to the object under it in the top view, kept inside that object's outline. `draw_states` shows it as a dashed
square. A cube in the slot is put at the slot's centre with the slot's angle (it can only sit in the notch). A cube on
the pan is good to about half a cube and its angle is not estimated; it is not a measurement.

## Assumptions

**Inventory**
- Exactly the five objects above, at most one of each name. Detection accounts for these names; it never
  discovers an object that is not in `INVENTORY`.
- Only a cube can be hidden on or in another object. A pan under the slot, or the slot on its side, is not handled.

**Shapes (`RIGID`, `SIZE_TOL`, `SOLIDITY_TOL`)**
- Templates are pixel sizes in cell 23's undistorted base view, measured on 2026-11-23, with 15 % tolerance on
  the sides and 0.05 on solidity. Another camera, calibration or object size needs new numbers.
- Objects lie flat on the glass. A slot or pan that does not (on its side, tilted on the rope) matches no template;
  it is then named by colour if it is the only leftover mask of its colour and about its size (`assign_inventory`).
  Its height stays the flat one, which is within 1 mm for the slot on its side.
- Two objects touching are assumed to still come out as two masks (true for cube against cube and cube against
  the frame in the test layouts).

**Colours**
- Shape proposes what a mask on the glass is, and colour has to agree (`assign_inventory`): the slot must be red,
  the pan green, and colour (a) tells the two cubes apart and rejects cube-sized grey robot parts. It also (b) seeds
  the rope (`_rope_points`), and (c) names a hidden cube in the top view (`_top_cube_name`).
  It also (d) drops pieces of the rope's top-view mask that are saturated like the blue frame (`_update`).
- All four are fixed HSV gates for today's paint and the cell's lighting. They are marked `ponytail:` in the code.

**Cell geometry (`CELLS`)**
- The glass rectangle and the base-to-top homography are per cell. Cell 23's homography is fitted on ten object
  centres, cell 20's on six points of one layout; both are good enough to drop a prompt, not to measure.
- Robot x, y come from `camera-to-robot-cr23.yaml` (existing) and `camera-to-robot-cr20.yaml` (ours, 2026-11-25:
  `scripts/debug/calibrate_cam_to_robot.py --target cube`, the blue cube held in the gripper at 36 poses).

**Use**
- `start` and `refresh` are called with the arm parked out of the way. An object behind the gripper in the top
  view at that moment cannot be prompted there.
- The caller decides when to pass the top image to `step`. The base view alone is enough for objects on the
  glass, because nothing can hide them from below.
- A status changes only by what the masks show. `refresh` is what adds an object that newly appears on the glass
  (a cube that fell off the pan); between refreshes such a cube stays `stacked` or becomes `unaccounted`.

## What is tested, and what is not

Tested by replaying recorded sweeps (`scripts/debug/perception_replay.py`), cell 23, 2026-11-23:
- Three layouts: objects apart; blue cube against the frame with the white cube touching it; white cube in the
  slot and blue cube on the pan. Every object got the right name and status in each.
- A stitched stream of four sweeps (2, 1, 2, 0) with a `refresh` at each join: statuses correct throughout,
  cubes that appear on the glass are added mid-stream, 258 steps at 64 ms median for both views.
- `tests/test_perception_inventory.py` checks the naming logic without a GPU.

Tested live on cell 23 (`scripts/debug/perception_live.py`, 2026-11-23, 55 loops, both views every step):
- Statuses stayed as at the start while the fingers pushed the pan with the blue cube on it. The cube left the
  pan and came to rest against the pan and the rope, off the glass; it stayed `stacked on pan` because its top
  mask still touches the pan's. "Stacked on X" means "off the glass and touching X", not "resting on X".
- Loop time, median: move request 15 ms, camera request 104 ms, undistortion 9 ms, tracking 61 ms. Both views on
  every step do not reliably fit a 200 ms period; the base view alone does.

Tested live on cell 20 (2026-11-25, colour prompts):
- Gripper flown over every object at top height: 146 frames, all five objects `on_glass` throughout, tracker 130 ms
  per frame for both views; cubes and rope within 0.6 mm of their first position, slot 1 mm, pan 3.5 mm (its first
  mask, taken with the gripper behind it, missed part of it).
- A ten-minute teleop session with pushes and blocked moves (`scripts/debug/teleop_filter.py`).

Not tested:
- A cube falling onto the glass between two `refresh` calls (the hand-off in `_handoff`).
- An object grasped, lifted or held; an object on the frame; an object leaving the workspace.
- The arm in front of a stacked object in the top view.
- Cell 20's colour prompts in other lighting; any cell other than 20 and 23.
