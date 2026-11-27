# CLAUDE.md — cloudgripper-wm

## Project Overview

This project trains a **world model** on the [CloudGripper](https://cloudgripper.org/) real cloud robotics facility at KTH Stockholm. CloudGripper is an open-source testbed with **32 small robot arm cells** controllable over HTTP APIs. The world model framework is [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) (`stable_worldmodel`), included as an editable git submodule under `third_party/stable-worldmodel`.

The goal is to collect real-robot interaction data in parallel across multiple robots, store it in a format compatible with `stable-worldmodel`, and train world models (JEPA-family: DINO-WM, LeWM, etc.) on that data.

## Repository Structure

```
cloudgripper-wm/
├── cloudgripper_wm/
│   ├── __init__.py
│   ├── world.py               # CloudGripperWorld wrapper (handles RobotPool + swm.World setup)
│   ├── envs/
│   │   ├── __init__.py        # gymnasium.register() for all task env IDs
│   │   ├── cloudgripper_env.py  # Core Gymnasium env wrapping CloudGripper API
│   │   ├── safe_cloudgripper_wrapper.py  # SafeCloudGripperWrapper — occupancy-based collision avoidance
│   │   ├── constants.py       # Shared X_RANGE / Y_RANGE / GRIPPER_RANGE clip constants
│   │   └── robot_pool.py     # Thread-safe singleton assigning robot names to env instances
│   ├── tasks/
│   │   ├── __init__.py        # TASK_REGISTRY dict, get_task() factory
│   │   ├── base.py            # Abstract Task base class + DEFAULT_HOME_POS
│   │   ├── cube_push.py       # CubePushTask (stub)
│   │   ├── cube_stack.py      # CubeStackTask (stub)
│   │   └── rope_manip.py      # RopeManipTask (stub)
│   ├── policies/
│   │   ├── __init__.py
│   │   ├── sticky_random_policy.py   # StickyRandomPolicy — holds a sampled action for N steps
│   │   └── geometric_trajectory_policy.py  # GeometricTrajectoryPolicy — circle/square/triangle in X-Y
│   ├── utils/
│   │   ├── cloudgripper_image_processor.py  # Camera undistortion / pixel<->world homography
│   │   ├── coordinate_converter.py          # Camera/robot coordinate conversion helpers
│   │   ├── get_finger_pos.py                # Gripper finger-tip geometry from (x,y,z,theta,w_grip)
│   │   ├── occupancy.py                     # World-frame occupancy heightmap + 3D collision checks
│   │   └── occupancy_viz.py                 # Shared 3D live-view helpers (sim + SafeCloudGripperWrapper)
│   ├── camera_params/          # Per-robot camera calibration YAMLs (cam-to-robot, base camera)
│   └── configs/               # (unused — configs live under scripts/data/config and scripts/train/config)
├── scripts/
│   ├── data/
│   │   ├── collect_cloudgripper.py  # Data collection entry point (Hydra, working)
│   │   ├── inspect_data.py          # Visualize collected Lance dataset as video
│   │   ├── extract_episode.py       # Extract a single episode to images + CSV
│   │   └── config/
│   │       ├── cloudgripper.yaml          # Default collection config (random policy)
│   │       ├── cloudgripper_geometric.yaml # Geometric trajectory policy config
│   │       └── launcher/
│   │           └── local.yaml             # W&B / launcher settings
│   ├── train/
│   │   ├── prejepa.py         # DINO-WM training entry point (Hydra)
│   │   └── config/
│   │       ├── prejepa.yaml   # CloudGripper PreJEPA training config (fully self-contained)
│   │       └── launcher/
│   │           └── local.yaml
│   ├── debug/
│   │   ├── test_connection.py        # Smoke test: images + base camera undistortion on real hardware
│   │   ├── teleop.py                 # Interactive teleop (no collision avoidance)
│   │   ├── teleop_safe.py            # Interactive teleop via SafeCloudGripperWrapper, matplotlib UI
│   │   ├── safety_sim.py              # Animate finger trajectories vs. synthetic heightmap — collision + wall-jam cases
│   │   ├── reset_objects.py           # Standalone test for SafeCloudGripperWrapper.push_objects_to_center()
│   │   ├── occupancy_demo.py         # Build heightmap from debug_base.jpg, save 3D viz
│   │   └── object_segmentation.py    # Quick HSV-based object mask tuning on debug_base.jpg
│   └── inspect_data.py        # Visualize collected Lance dataset as video
├── tests/
│   ├── conftest.py            # Shared fixtures (empty — not yet written)
│   ├── test_env_base.py       # (empty — not yet written)
│   ├── test_env_tasks.py      # (empty — not yet written)
│   ├── test_tasks.py          # (empty — not yet written)
│   ├── test_robot_pool.py     # (empty — not yet written)
│   └── test_registration.py   # (empty — not yet written)
├── third_party/
│   ├── cloudgripper-api/      # Git submodule — has local pyproject.toml for uv editable install
│   └── stable-worldmodel/     # Git submodule (editable dep)
├── data/                      # Collected datasets (gitignored)
├── pyproject.toml
├── uv.lock
└── CLAUDE.md                  # This file
```

## Tech Stack & Dependencies

- **Python ≥ 3.10**, managed with **uv** (not pip/poetry)
- **stable-worldmodel[env,train]** — editable from `third_party/stable-worldmodel`
  - Provides: `swm.World`, `swm.data.load_dataset`, `EnvPool`, `MegaWrapper`, `Policy`, `CEMSolver`, `WorldModelPolicy`, etc.
- **cloudgripper-api** — the `GripperRobot` class from `client.cloudgripper_client`
  - Repo: https://github.com/cloudgripper/cloudgripper-api
  - Installed as git submodule under `third_party/cloudgripper-api/` with a local `pyproject.toml` added for uv editable install (the upstream repo has no build system)
  - Also has `GripperRobotMock` in `client.cloudgripper_client_mock` for testing without hardware
- **PyTorch** — for world model training
- **Gymnasium** — env interface (`gymnasium.Env`)
- **datasets** (HuggingFace) — listed as a dependency

### Build commands

```bash
uv sync                    # install all deps from lockfile
uv run python <script>     # run anything in the venv
uv run pytest tests/       # run tests
```

### Scripts

All scripts use Hydra — any config key can be overridden on the CLI.

```bash
# Data collection — `episodes` is the total target (resumes automatically)
uv run python scripts/data/collect_cloudgripper.py output=data/my_run
uv run python scripts/data/collect_cloudgripper.py output=data/my_run robots=[robot1,robot2] episodes=100
uv run python scripts/data/collect_cloudgripper.py --config-name cloudgripper_geometric output=data/shapes

# Training (dataset_name is required)
uv run python scripts/train/prejepa.py dataset_name=$(pwd)/data/my_run/my_run.lance
uv run python scripts/train/prejepa.py dataset_name=$(pwd)/data/my_run/my_run.lance trainer.max_epochs=200

# Real-robot smoke test
uv run python scripts/debug/test_connection.py      # defaults to robot23
uv run python scripts/debug/teleop_safe.py --robot robot5  # interactive teleop w/ collision avoidance

# Data inspection
uv run python scripts/data/inspect_data.py data/my_run/my_run.lance
uv run python scripts/data/inspect_data.py data/my_run/my_run.lance --save-dir /tmp/videos
```

### Dev robot

- **robot20** is the development robot since 2026-11-25 (working gripper and height axis; its camera files and the
  filter's numbers are ours). robot23's gripper is off its servo, robot22's height axis does not respond.
  Older scripts still default to robot23.
- On robot20 the gripper is closed at 0.1 and open at 0.825; x beyond 0.88 is not visible from below.
- Set the token before any real-robot code: `export CLOUDGRIPPER_TOKEN=<token>`
  - Do NOT commit the token to any file — keep it in your shell environment only
- Quick smoke test on real hardware: `uv run python scripts/debug/test_connection.py`
- Interactive teleop with collision avoidance: `uv run python scripts/debug/teleop_safe.py`
  - Defaults to robot23
  - Pass `--help` for all options

## CloudGripper Robot API

The robot is controlled via `GripperRobot(robot_name, token)` from `cloudgripper-api`. Key methods:

### Actions (all arguments normalized 0–1 unless noted)
- `robot.move_xy(x, y)` — move to (x, y) position in work area, both in [0, 1]
- `robot.move_z(z)` — vertical position, z in [0, 1]
- `robot.rotate(angle)` — rotation in degrees [0, 180]
- `robot.move_gripper(val)` — gripper aperture, 0 = closed, 1 = open
- `robot.gripper_open()` / `robot.gripper_close()` — convenience
- `robot.step_forward()` / `step_backward()` / `step_left()` / `step_right()` — discrete steps

### Observations
- `robot.get_state()` → `(state_dict, timestamp)` — robot state (positions, gripper, etc.)
- `robot.getImageTop()` → `(image_ndarray, timestamp)` — top-down camera (primary for world model)
- `robot.getImageBase()` → `(image_ndarray, timestamp)` — base/bottom camera

### Authentication
- Token via env var `CLOUDGRIPPER_TOKEN`
- Robot names are `"robot1"` through `"robot32"`

### Mock client
- `GripperRobotMock` from `client.cloudgripper_client_mock` — same interface, returns blank frames, configurable `failure_rate`

## stable-worldmodel API (Key Concepts)

### World (main entry point)
```python
import stable_worldmodel as swm
import cloudgripper_wm.envs  # triggers gymnasium.register() for all env IDs

# Task-agnostic — for world model data collection & training
world = swm.World(
    env_name="cloudgripper/Gripper-v0",    # task-agnostic (task=None)
    num_envs=8,                             # parallel envs (= parallel robots)
    image_shape=(64, 64),                   # resize target for pixels
    max_episode_steps=100,
)

# Task-specific — for RL baselines & MPC evaluation
world = swm.World(
    env_name="cloudgripper/CubePush-v0",   # task="cube_push" injected via kwargs
    num_envs=8,
    image_shape=(64, 64),
    max_episode_steps=100,
)
```
- `World` creates an `EnvPool` of `num_envs` envs, each wrapped by `MegaWrapper`
- `MegaWrapper` expects `"pixels"` key in the info dict — this is the primary image observation
- `world.set_policy(policy)` → attach a policy
- `world.collect(path, episodes=N, seed=0)` → roll out and save dataset
- `world.evaluate(episodes=N)` → run evaluation, returns `{"success_rate": ...}`

### Info convention
- All tensor values in `world.infos` have shape `(num_envs, 1, ...)`
- `world.infos["pixels"]` → `(num_envs, 1, H, W, C)`

### Data formats
- **Lance** (default), **HDF5**, **folder**, **video**, **lerobot**
- `swm.data.load_dataset("path.lance", num_steps=16)` — autodetects format
- Custom formats can be registered

### Policy protocol
```python
class Policy:
    def get_action(self, infos: dict) -> np.ndarray:
        """Return actions of shape (num_envs, action_dim)."""
        ...
    def set_env(self, envs):
        """Called by World.set_policy() to configure policy for these envs."""
        ...
```
- `RandomPolicy` is provided by `stable_worldmodel.policy`

### Training reference
- `scripts/train/prejepa.py` in stable-worldmodel reproduces DINO-WM
- `scripts/train/gcivl.py` implements goal-conditioned RL baselines
- Training uses Hydra for config

## CloudGripperEnv Design Decisions

### Observation space

**`observation_space`** only contains `"state"` — images are NOT in the obs dict:
- **`"state"`**: float32 vector `[x, y, z, rotation_norm, gripper]`, all in [0, 1]. This is `self._target_pos` (commanded target), not the actual robot position which may lag.

**Why images are not in obs:** `MegaWrapper` stacks `AddPixelsWrapper` → `EverythingToInfoWrapper`. `AddPixelsWrapper` calls `env.render()` and writes `"pixels"` into info. `EverythingToInfoWrapper` then asserts obs keys are not already in info before merging them. If obs had `"pixels"`, this assertion would fail. So images flow through `render()`, not obs.

**Where images end up in the dataset:**
- `env.render()` → top camera → `AddPixelsWrapper` writes `"pixels"` to info → stored in dataset as `"pixels"` (JPEG-encoded, 64×64 after MegaWrapper resize)
- `info["pixels_base"]` → base camera → stored in dataset as `"pixels_base"` (JPEG-encoded, native 480×640)
- `world.infos["pixels"]` has shape `(num_envs, 1, 64, 64, 3)` during collection

### Action space (delta)
- `Box(-max_delta, max_delta, shape=(5,), dtype=float32)` → `[Δx, Δy, Δz, Δrotation, Δgripper]`
- **Delta actions, not absolute.** The env maintains an internal `self._target_pos: np.ndarray` of shape `(5,)` representing `[x, y, z, rotation_norm, gripper]`, all in [0, 1].
- Each step: `self._target_pos = np.clip(self._target_pos + action, 0.0, 1.0)`, then the clipped absolute values are sent to the robot API.
- After the full [0,1] clip, `x`, `y`, and `gripper` are further clipped to the ranges defined in `cloudgripper_wm/envs/constants.py` (`X_RANGE = (0.175, 0.825)`, `Y_RANGE = (0.175, 0.825)`, `GRIPPER_RANGE = (0.0, 1.0)`). `SafeCloudGripperWrapper` applies the same clamps to its candidate position so the two stay in sync.
- `max_delta` is a configurable constructor parameter (default ~0.05–0.1). This caps how far the robot can move per step, producing smooth trajectories.
- The CloudGripper API only accepts absolute coordinates, so the env is responsible for the delta-to-absolute conversion. The robot API never sees deltas.

#### Delta-to-absolute mapping
```python
# In step():
self._target_pos = np.clip(self._target_pos + action, 0.0, 1.0)
x, y, z, rot_norm, grip = self._target_pos
self.robot.move_xy(float(x), float(y))
self.robot.move_z(float(z))
self.robot.rotate(float(rot_norm) * 180.0)   # [0,1] → [0°,180°]
self.robot.move_gripper(float(grip))          # 0=closed, 1=open
```

#### Internal position tracking
- `self._target_pos` is the *commanded* target, not the actual robot position (which may lag due to movement time).
- On `reset()`, set `self._target_pos` to `self.task.home_pos()` if task is set, otherwise `DEFAULT_HOME_POS` `[0.1, 0.1, 1.0, 0.0, 0.0]` (near-corner xy, top z, 0° rotation, gripper closed). Note: z=1.0 means arm up — calibrated from real robot testing.
- The `"state"` in the observation space reports `self._target_pos` (the intended target). Optionally, a `"state_actual"` key can store the readback from `robot.get_state()` if the caller wants to compare.
- Do NOT re-sync `_target_pos` from `get_state()` every step — it adds latency and the robot may not have reached the target yet. Only re-sync on `reset()` if needed.

### Step timing
- Real robot has HTTP latency + physical movement time
- After sending commands, sleep for a configurable `dwell_time` (default ~0.5s) before reading observations
- This means data collection is slow compared to simulation — parallel robots are essential
- The `dwell_time` should be long enough that the robot approximately reaches `_target_pos` before the next observation is captured. With small `max_delta` values this is naturally satisfied.

### Reset behavior
- Move to home position: `self.task.home_pos()` if task is set, otherwise `DEFAULT_HOME_POS` `[0.1, 0.1, 1.0, 0.0, 0.0]` (near-corner xy, top z, 0° rotation, gripper closed)
- Set `self._target_pos` accordingly
- Send absolute commands to robot: `move_xy`, `move_z`, `rotate`, `gripper_open`
- Object rearrangement on the workspace is manual/out-of-scope for now
- Episodes are fixed-length (truncation via `max_episode_steps`). Early termination only possible when a task is set and `task.check_terminated()` returns True.

### Multi-robot parallelism via RobotPool

**RobotPool is always required** — even for a single robot. Call `RobotPool.configure(["robot23"])` before creating any envs. `CloudGripperEnv.__init__` calls `RobotPool.acquire()` and `close()` calls `RobotPool.release()`. The interface is identical for 1 or N robots.

```python
from cloudgripper_wm.envs.robot_pool import RobotPool

RobotPool.configure(["robot23"])            # 1 robot
RobotPool.configure(["robot1", "robot2"])   # 2 robots — identical call site
```

- `swm.World` creates `num_envs` copies of the env via `gym.make` with identical kwargs
- Constraint: `num_envs == len(robot_names)` passed to `RobotPool.configure()`
- Use `CloudGripperWorld` (see below) to handle this automatically

### CloudGripperWorld wrapper (`cloudgripper_wm/world.py`)

Thin wrapper around `swm.World` that handles RobotPool configuration and token resolution:

```python
from cloudgripper_wm.world import CloudGripperWorld
from stable_worldmodel.policy import RandomPolicy

# 1 robot
world = CloudGripperWorld(robot_names=["robot23"])

# 8 robots — identical interface
world = CloudGripperWorld(robot_names=["robot1", "robot2", ..., "robot8"])

world.set_policy(RandomPolicy())
world.collect("data/collect.lance", episodes=100)
world.close()
```

Constructor kwargs: `robot_names`, `env_name` (default `"cloudgripper/Gripper-v0"`), `image_shape` (default `(64, 64)`), `max_episode_steps`, `token` (falls back to `CLOUDGRIPPER_TOKEN` env var), `use_mock`, `dwell_time`, `max_delta`, `safety` (optional dict — see [SafeCloudGripperWrapper (Collision Avoidance)](#safecloudgrippercwrapper-collision-avoidance) below). Any extra kwargs are forwarded to `swm.World` / `gym.make`.

When `safety` is set, `CloudGripperWorld` adds `SafeCloudGripperWrapper(live_view=False, **safety)` as an `extra_wrapper` — applied **outside** `MegaWrapper`, so the recorded `"action"` column reflects the actually-executed (post-safety-filter) action, not the policy's raw output.

### Lance dataset schema

Collected datasets are stored in Lance format. Each row is one step:

| Column | Type | Description |
|--------|------|-------------|
| `episode_idx` | int32 | Episode number |
| `step_idx` | int32 | Step within episode |
| `pixels` | bytes | Top camera JPEG, 64×64 after MegaWrapper resize |
| `pixels_base` | bytes | Base camera JPEG, native 480×640 |
| `state` | float[5] | `[x, y, z, rot_norm, gripper]` — commanded target pos |
| `action` | float[5] | `[Δx, Δy, Δz, Δrot, Δgrip]` — delta action taken (post-safety-filter if `SafeCloudGripperWrapper` is enabled) |
| `reward` | float[1] | Always 0.0 for task-agnostic collection |
| `terminated` | float[1] | |
| `truncated` | float[1] | |
| `id` | float[1] | Episode UUID (from EverythingToInfoWrapper) |

Decode images: `cv2.imdecode(np.frombuffer(row["pixels"], np.uint8), cv2.IMREAD_COLOR)`

## SafeCloudGripperWrapper (Collision Avoidance)

`cloudgripper_wm/envs/safe_cloudgripper_wrapper.py` wraps `CloudGripperEnv` (or a `MegaWrapper`-wrapped env, when used via `CloudGripperWorld(safety=...)`) and blocks actions that would press the gripper down onto an object.

### Pipeline

1. **Heightmap** — after every `reset()`/`step()`, `utils/occupancy.build_height_map()` builds a world-frame occupancy heightmap from the base camera image (`info["pixels_base"]`). Objects are detected via HSV thresholding — **default range assumes GREEN objects** (`hsv_lower=(35,60,40)`, `hsv_upper=(90,255,255)`); other colors are not detected/avoided. Each occupied cell is extruded to a fixed `height` (constructor param, default `0.35`) — this is an *assumed* object height, not measured.
2. **Finger prediction** — `utils/occupancy_viz.pose_to_fingers(pose)` converts `[x, y, z, rot_norm, gripper]` → left/right finger-tip world positions via `utils/get_finger_pos.get_finger_pos(x, y, z, theta, w_grip)` (note argument order: `theta` before `w_grip`).
3. **Collision check** (`SafeCloudGripperWrapper._check_collision`), run on every `step()` before forwarding the action:
   - For each finger, `utils/occupancy.check_collision(start, end, hmap, surface="top")` checks whether the straight-line path from the current to the candidate finger position **crosses** the heightmap's top surface (a sign change in `z - top`). This allows lateral pushes (entering/staying inside a column without crossing top) but blocks descending through it.
   - **Edge case — already inside the column**: the top-crossing check only fires once, on the initial crossing. If a finger has *already* entered an occupied column (e.g. via a lateral move, or because `height` overestimates the real object so there's a "buffer" zone below the assumed top but above the real surface), subsequent downward steps never cross `top` again and would pass unchecked. `utils/occupancy.is_inside_occupancy(point, hmap)` catches this: if a finger's *current* position is inside an occupied column (`top > 0` and `z <= top`) **and** the candidate position has a lower `z`, the action is blocked too. Lateral and upward moves while inside remain allowed.
   - **Wall-jam guard**: the workspace has a low wall around its edge (outside the gripper's reach), and an object pushed hard against it can pop out over the wall — also risking gripper damage from sustained contact force. This check only applies when the *candidate* finger position is actually in contact with the object — `is_inside_occupancy(p1, hmap)` (i.e. `(x,y)` over an occupied cell **and** `z <= top`) — so a finger merely flying over an object's footprint at a safe height is never flagged. When in contact, `utils/occupancy.object_near_wall(point_xy, hmap, bounds, margin)` finds the connected occupied component (via `cv2.connectedComponents`) at the finger's candidate `(x, y)` cell and checks whether that *object's full bounding extent* (not just the contact cell) comes within `wall_margin` of a `WORKSPACE_BOUNDS` edge — checking the whole object matters because a wide object can already be jammed against the wall on its far side while the finger contacts it from the near side, well outside the margin band. If an edge is returned, the move is blocked only if the finger is moving toward (or holding at) that edge along the relevant axis — `sign * (p1[axis] - p0[axis]) >= 0`; retreating away from the wall is always allowed. A small jam/gap is fine — only *sustained pushing into* the wall is blocked.
4. **Enforcement** — if either finger's check fails, the action is replaced with `np.zeros_like(action)` (target position held in place) and a warning is printed. The wrapper's `step()` calls `self.env.step(action)` with this *corrected* action — so when used as an `extra_wrapper` (outside `MegaWrapper`), the dataset's `"action"` column reflects what was actually executed, not the rejected action.

### Constructor kwargs

`SafeCloudGripperWrapper(env, cell_size=0.01, height=0.35, live_view=False, grid_cells=25, wall_margin=0.21, reset_objects_every=10, cooldown_every=10, cooldown_time=30.0, **height_map_kwargs)` — `height_map_kwargs` are forwarded to `build_height_map` (e.g. `hsv_lower`/`hsv_upper`, `dilate_cells`). `wall_margin` is the distance (in `_target_pos`/occupancy world units) from a `WORKSPACE_BOUNDS` edge within which an object is considered "against the wall" for the wall-jam guard. `live_view=True` opens a `LiveOccupancyView` (interactive 3D matplotlib plot) — only intended for single-robot debugging (`scripts/debug/teleop_safe.py`), not parallel data collection. `reset_objects_every` controls the periodic workspace reset — see below; set to `0` to disable. `cooldown_every`/`cooldown_time` control a periodic robot cooldown — see below; set `cooldown_every=0` to disable.

### Qt/cv2 conflict

`utils/occupancy_viz.LiveOccupancyView.__init__` lazily imports matplotlib and pops `QT_QPA_PLATFORM_PLUGIN_PATH` (cv2 and matplotlib ship incompatible bundled Qt plugins) — this only happens when `live_view=True`. Scripts that use both `cv2.imshow` (e.g. `CloudGripperEnv(show_display=True)`) and matplotlib (e.g. `teleop_safe.py`) must do this fixup themselves before importing `matplotlib.pyplot`.

### Workspace reset routine — `push_objects_to_center`

`SafeCloudGripperWrapper.push_objects_to_center(push_amount=0.25, n_sweeps=7, transit_z=0.5, push_z=0.2, dwell_time=1.0, on_step=None)` sweeps the gripper around all four edges of `WORKSPACE_BOUNDS`, pushing any objects near the walls back toward the center. For each side, it visits `n_sweeps` evenly spaced positions along that edge (gripper closed, `rot_norm=0`); at each position it moves to the edge at `transit_z`, descends to `push_z`, pushes `push_amount` inward, then retreats to `transit_z` before moving on. Each of these is a single absolute move via the internal `_move_to()` helper — one `step()` call (one robot move + one dwell) regardless of `max_delta`, since `CloudGripperEnv._send_absolute()` already sends absolute per-axis commands. Collision/wall-jam checks run on the full move's finger trajectory, so this is not weaker than chunking into small steps. A blocked move (e.g. an object already sitting where the gripper would descend) is logged and skipped rather than retried. The optional `on_step(obs, info)` callback fires after every `step()` — used by `reset_objects.py` to drive a live camera view / video recorder.

For the duration of this routine, `CloudGripperEnv._restrict_xy` (an instance flag, default `True`) is temporarily set to `False` so `step()` and the wrapper's collision-check candidate position skip the `X_RANGE`/`Y_RANGE` clamps — edges/pushes need to reach the full `WORKSPACE_BOUNDS` to retrieve objects near the walls. `CloudGripperEnv.dwell_time` is also temporarily overridden to `dwell_time` (default `1.0`s, larger than the typical per-step dwell), since these moves cover larger distances and need more settling time. Both are restored in a `finally` block.

Standalone test: `scripts/debug/reset_objects.py` (`--robot`, `--push-amount`, `--n-sweeps`, `--use-mock`, `--live-view`, etc.). `--live-view` shows both the 3D occupancy/finger view and a top+base camera feed (matplotlib, updated after every move); `--video-dir DIR` (requires `--live-view`) additionally records both views to `camera.mp4` / `occupancy.mp4` via `matplotlib.animation.FFMpegWriter`.

### Periodic workspace reset in `reset()`

`SafeCloudGripperWrapper` tracks an internal `_episode_count`, incremented on every `reset()`. When `_episode_count % reset_objects_every == 0` (default every 10th episode), `reset()` calls `push_objects_to_center()` (with default args) after the normal home-position reset, then calls `self.env.reset(**kwargs)` again to return to home before returning. Set `reset_objects_every=0` to disable. This runs with whatever `live_view`/heightmap settings the wrapper was constructed with, so the same collision/wall-jam checks apply during the periodic reset as during normal stepping.

### Periodic robot cooldown in `reset()`

Independently of the workspace reset above, `reset()` also checks `_episode_count % cooldown_every == 0` (default every 10th episode) and, if true, calls `time.sleep(cooldown_time)` (default `30.0`s) before returning — giving the robot hardware a rest period on a fixed cadence. `cooldown_every` and `cooldown_time` are separate constructor kwargs from `reset_objects_every`/`push_objects_to_center`'s args, so the two periodic behaviors can run on different schedules (or either can be disabled independently by setting its `_every` param to `0`). Both checks use the same `_episode_count`, so with the defaults (`10` and `10`) they coincide on the same episodes.

## Task System

### Two usage paths

The project has two distinct usage modes that share the same base env but differ in whether a task is attached:

1. **Task-agnostic (world model training):** `cloudgripper/Gripper-v0` — no `Task` object, always returns `reward=0.0`, `terminated=False`, no `"success"` key in info. Used for self-supervised WM data collection and training (JEPA, DINO-WM, LeWM). The world model learns environment dynamics regardless of what objects are on the table.

2. **Task-specific (RL baselines / MPC evaluation):** `cloudgripper/CubePush-v0`, `cloudgripper/CubeStack-v0`, `cloudgripper/RopeManip-v0` — a `Task` object provides reward, success, and termination logic. Used for RL training and for `world.evaluate()` with MPC solvers.

**The physical cell setup is entirely manual** — someone places a cube, rope, etc. on the workspace. Nothing about the task is controllable from code. The `Task` object only interprets what the cameras see and defines what "success" means.

### Architecture

The env's `task` parameter is `Optional[str]`, defaulting to `None`. When `None`, all task hooks are skipped and the env is purely task-agnostic. When set, the env looks up a `Task` instance from the registry and delegates to it.

```
CloudGripperEnv(task=None)              CloudGripperEnv(task="cube_push")
  │                                       │
  ├── self.robot: GripperRobot            ├── self.robot: GripperRobot
  ├── self._target_pos: np.ndarray        ├── self._target_pos: np.ndarray
  └── self.task = None                    └── self.task: CubePushTask
       → reward=0.0 always                     ├── compute_reward(obs, action, info) → float
       → terminated=False always               ├── check_success(obs, info) → bool
       → no "success" in info                  ├── check_terminated(obs, info) → bool
                                               ├── get_task_info(obs) → dict
                                               └── home_pos() → np.ndarray
```

### Task base class (`tasks/base.py`)
```python
from abc import ABC, abstractmethod
import numpy as np

DEFAULT_HOME_POS = np.array([0.1, 0.1, 1.0, 0.0, 0.0], dtype=np.float32)

class Task(ABC):
    """Base class for CloudGripper tasks. Subclass to define reward and success."""

    @abstractmethod
    def compute_reward(self, obs: dict, action: np.ndarray, info: dict) -> float:
        """Return scalar reward for this transition."""
        ...

    @abstractmethod
    def check_success(self, obs: dict, info: dict) -> bool:
        """Return True if the task goal has been achieved."""
        ...

    def check_terminated(self, obs: dict, info: dict) -> bool:
        """Return True if episode should end early (not truncation). Default: False."""
        return False

    def get_task_info(self, obs: dict) -> dict:
        """Return extra keys to merge into step info (stored in dataset). Default: empty."""
        return {}

    def home_pos(self) -> np.ndarray:
        """Task-specific home position [x, y, z, rot_norm, gripper]. Default: center/top/open."""
        return DEFAULT_HOME_POS.copy()
```

### Current tasks

| Task key       | Class            | Gymnasium ID                    | Description |
|----------------|------------------|---------------------------------|-------------|
| *(none)*       | —                | `cloudgripper/Gripper-v0`      | **Task-agnostic.** For WM data collection and training. |
| `cube_push`    | `CubePushTask`   | `cloudgripper/CubePush-v0`     | Push a cube to a target region on the workspace |
| `cube_stack`   | `CubeStackTask`  | `cloudgripper/CubeStack-v0`    | Stack one cube on top of another |
| `rope_manip`   | `RopeManipTask`  | `cloudgripper/RopeManip-v0`    | Manipulate a rope into a target configuration |

**More tasks will be added as new physical setups are deployed.** To add a task: create a new `Task` subclass in `cloudgripper_wm/tasks/`, register it in `TASK_REGISTRY`, and add a Gymnasium `register()` call in `envs/__init__.py`.

### Initial implementations

For the initial phase, all three task classes can start as **stubs** that return `reward=0.0` and `success=False`. The reward/success methods become important later when running RL baselines or MPC evaluation via `world.evaluate()`. The task-agnostic `cloudgripper/Gripper-v0` path doesn't need them at all.

### Task registry (`tasks/__init__.py`)
```python
from cloudgripper_wm.tasks.base import Task
from cloudgripper_wm.tasks.cube_push import CubePushTask
from cloudgripper_wm.tasks.cube_stack import CubeStackTask
from cloudgripper_wm.tasks.rope_manip import RopeManipTask

TASK_REGISTRY: dict[str, type[Task]] = {
    "cube_push": CubePushTask,
    "cube_stack": CubeStackTask,
    "rope_manip": RopeManipTask,
}

def get_task(task_name: str | None) -> Task | None:
    """Instantiate a task by name. Returns None if task_name is None (task-agnostic mode)."""
    if task_name is None:
        return None
    return TASK_REGISTRY[task_name]()
```

### How the env uses the task
```python
# In CloudGripperEnv.__init__(self, ..., task: str | None = None):
from cloudgripper_wm.tasks import get_task
from cloudgripper_wm.tasks.base import DEFAULT_HOME_POS
self.task = get_task(task)  # None for task-agnostic, Task instance otherwise

# In step():
if self.task is not None:
    reward = self.task.compute_reward(obs, action, info)
    terminated = self.task.check_terminated(obs, info)
    info["success"] = self.task.check_success(obs, info)
    info.update(self.task.get_task_info(obs))
else:
    reward = 0.0
    terminated = False

# In reset():
if self.task is not None:
    self._target_pos = self.task.home_pos().copy()
else:
    self._target_pos = DEFAULT_HOME_POS.copy()
```

### Gymnasium registration (`envs/__init__.py`)
```python
from gymnasium.envs.registration import register

# Task-agnostic base env (for world model data collection & training)
register(
    id="cloudgripper/Gripper-v0",
    entry_point="cloudgripper_wm.envs.cloudgripper_env:CloudGripperEnv",
    max_episode_steps=100,
    # task defaults to None inside CloudGripperEnv
)

# Task-specific envs (for RL baselines & MPC evaluation)
_TASKS = {
    "cloudgripper/CubePush-v0": "cube_push",
    "cloudgripper/CubeStack-v0": "cube_stack",
    "cloudgripper/RopeManip-v0": "rope_manip",
}

for env_id, task_name in _TASKS.items():
    register(
        id=env_id,
        entry_point="cloudgripper_wm.envs.cloudgripper_env:CloudGripperEnv",
        max_episode_steps=100,
        kwargs={"task": task_name},
    )
```

### Usage
```python
import stable_worldmodel as swm
import cloudgripper_wm.envs  # triggers registration

# World model training — task-agnostic, no reward needed
world = swm.World("cloudgripper/Gripper-v0", num_envs=8, image_shape=(64, 64))

# RL baseline — task-specific, with reward and success
world = swm.World("cloudgripper/CubePush-v0", num_envs=8, image_shape=(64, 64))
```

## Implementation Status

| Step | Status | Notes |
|------|--------|-------|
| Task base class + stubs | ✅ Done | `tasks/base.py`, stubs in `cube_push/stack/rope_manip.py` |
| `CloudGripperEnv` | ✅ Done | Tested on robot23 |
| `RobotPool` | ✅ Done | Always required, even for 1 robot |
| Gymnasium registration | ✅ Done | All 4 env IDs registered |
| `CloudGripperWorld` wrapper | ✅ Done | `cloudgripper_wm/world.py` |
| Data collection script | ✅ Done | `scripts/data/collect_cloudgripper.py` — Hydra-based, resume-safe, config saved alongside dataset |
| Data inspection | ✅ Done | `scripts/data/inspect_data.py` — video player + action overlay |
| Training script | ✅ Done | `scripts/train/prejepa.py`, DINO-WM via stable-worldmodel's prejepa pipeline |
| `StickyRandomPolicy` | ✅ Done | `policies/sticky_random_policy.py` — action persistence + Gaussian noise |
| `GeometricTrajectoryPolicy` | ✅ Done | `policies/geometric_trajectory_policy.py` — circle/square/triangle in X-Y, sticky-random on other DoFs |
| Live display thread | ✅ Done | `CloudGripperEnv(show_display=True)` shows top camera in a background thread via `cv2.imshow` |
| `SafeCloudGripperWrapper` | ✅ Done | `envs/safe_cloudgripper_wrapper.py` — occupancy-heightmap collision check (`surface="top"`), blocks actions that would collide, optional live 3D view (`scripts/debug/teleop_safe.py`). **Assumes GREEN objects** (default HSV range in `utils/occupancy.py`); other colors are not detected/avoided. |
| Tests | ❌ Not written | Test files exist but are empty |
| Reward & success implementations | ❌ Deferred | Only needed for RL/MPC evaluation |

## Training

### PreJEPA vs DINO-WM

**PreJEPA** is the training algorithm — a joint-embedding predictive architecture that learns to predict future embeddings from past ones (I-JEPA/V-JEPA family). The script is named `prejepa.py` after the algorithm.

**DINO-WM** is a specific instantiation of PreJEPA that uses a DINOv2 encoder as the backbone. With `backbone.name: dinov2_small` (the default), `prejepa.py` trains DINO-WM. Swapping to a different backbone (MAE, ResNet, etc.) still uses the same PreJEPA training loop.

Each world model variant gets its own script: `scripts/train/prejepa.py` for PreJEPA-family models, `scripts/train/lewm.py` for LeWM when added, etc.

### Running

`scripts/train/prejepa.py` wraps stable-worldmodel's PreJEPA pipeline with a fully self-contained CloudGripper config at `scripts/train/config/prejepa.yaml` — no modifications to the stable-worldmodel submodule are needed.

```bash
uv run python scripts/train/prejepa.py dataset_name=$(pwd)/data/collect.lance
```

### Training config (`scripts/train/config/prejepa.yaml`)

Key CloudGripper-specific settings (all others match stable-worldmodel defaults):

| Key | Value | Reason |
|-----|-------|--------|
| `frameskip` | `1` | No temporal skip — robot data is already at action frequency |
| `wm.encoding.action` | `10` | Embed 5-dim delta actions into 10-dim space |
| `wm.encoding.state` | `10` | Embed 5-dim commanded target into 10-dim space |
| `backbone.name` | `dinov2_small` | DINOv2 backbone → this makes it DINO-WM specifically |
| `image_size` | `224` | DINOv2 input size (images upscaled 64→224 internally) |
| `trainer.max_epochs` | `100` | Default training length |

The world model trains on `"pixels"` (top camera) and uses `"action"` + `"state"` as conditioning. No reward signal — fully self-supervised.

### Decoupling from stable-worldmodel

The training logic imports helpers from `third_party/stable-worldmodel/scripts/train/prejepa.py` at runtime via `sys.path`, but makes **zero modifications to any file in the submodule**. The stable-worldmodel submodule is kept at its upstream state.

## Tests

All tests use `GripperRobotMock` so they run without hardware or network. Run with `uv run pytest tests/`.

### Test structure

```
tests/
├── conftest.py            # Shared fixtures
├── test_env_base.py       # Task-agnostic env (cloudgripper/Gripper-v0)
├── test_env_tasks.py      # Task-specific envs (CubePush, CubeStack, RopeManip)
├── test_tasks.py          # Task classes in isolation
├── test_robot_pool.py     # RobotPool thread-safety
└── test_registration.py   # Gymnasium registration + swm.World integration
```

### `conftest.py` — shared fixtures

- `mock_env(task=None)` — factory fixture that creates a `CloudGripperEnv` using `GripperRobotMock`, with configurable `task` parameter. Handles `env.close()` in teardown.
- `mock_env_base()` — convenience fixture: `mock_env(task=None)`.
- `mock_env_cube_push()` — convenience fixture: `mock_env(task="cube_push")`. Same pattern for other tasks.
- Fixtures should patch the robot client import so `CloudGripperEnv` uses `GripperRobotMock` instead of `GripperRobot`.

### `test_env_base.py` — task-agnostic env

These tests verify the core env contract without any task logic:

- **`test_reset_returns_valid_obs`** — call `env.reset()`, check that returned `obs` contains `"pixels"` (uint8, correct shape), `"pixels_base"` (uint8, correct shape), and `"state"` (float32, shape `(5,)`, values in [0, 1]).
- **`test_reset_sets_home_position`** — after `reset()`, `env._target_pos` equals `DEFAULT_HOME_POS`.
- **`test_step_returns_gymnasium_tuple`** — call `env.step(action)`, check it returns `(obs, reward, terminated, truncated, info)` with correct types.
- **`test_reward_always_zero`** — step with arbitrary actions, assert `reward == 0.0` every time.
- **`test_terminated_always_false`** — step with arbitrary actions, assert `terminated is False` every time.
- **`test_no_success_key_in_info`** — after step, assert `"success"` not in `info`.
- **`test_delta_action_clipping`** — set `_target_pos` near boundary (e.g. `[0.99, 0.01, ...]`), step with a delta that overshoots, verify `_target_pos` is clipped to [0, 1] and doesn't go out of bounds.
- **`test_delta_action_accumulation`** — step with known deltas, verify `_target_pos` updates correctly across multiple steps (cumulative).
- **`test_action_space_shape_and_bounds`** — `env.action_space` is `Box` with correct shape `(5,)`, symmetric bounds `[-max_delta, max_delta]`.
- **`test_observation_space_structure`** — `env.observation_space` is a `Dict` with the expected keys and sub-spaces.
- **`test_reset_after_steps`** — step a few times, then reset, verify `_target_pos` is back to home and obs is valid.
- **`test_episode_truncation`** — step `max_episode_steps` times, verify `truncated=True` on the last step (if env handles this, otherwise verify `World` handles it).

### `test_env_tasks.py` — task-specific envs

These tests verify that task injection works correctly. Run the same structural checks but with task-specific behavior:

- **`test_task_env_has_success_key`** — for each task env (`cube_push`, `cube_stack`, `rope_manip`): step and assert `"success"` IS present in `info`.
- **`test_task_env_delegates_reward`** — step and verify reward comes from `task.compute_reward()` (for stubs this is 0.0, but the code path is exercised).
- **`test_task_env_delegates_terminated`** — step and verify terminated comes from `task.check_terminated()`.
- **`test_task_env_home_position`** — reset and verify `_target_pos` matches `task.home_pos()`, not `DEFAULT_HOME_POS` (they may be the same for stubs, but this tests the code path).
- **`test_task_info_merged`** — verify that `task.get_task_info()` dict is merged into step info.
- **Parametrize over all tasks** — use `@pytest.mark.parametrize("task", ["cube_push", "cube_stack", "rope_manip"])` to avoid duplicating test functions.

### `test_tasks.py` — task classes in isolation

Test the `Task` subclasses directly without the env:

- **`test_task_registry_contains_all_tasks`** — `TASK_REGISTRY` has keys `"cube_push"`, `"cube_stack"`, `"rope_manip"`.
- **`test_get_task_returns_none_for_none`** — `get_task(None)` returns `None`.
- **`test_get_task_returns_instance`** — `get_task("cube_push")` returns a `CubePushTask` instance (and similarly for others).
- **`test_get_task_raises_on_unknown`** — `get_task("nonexistent")` raises `KeyError`.
- **`test_home_pos_shape_and_bounds`** — for each task, `task.home_pos()` returns float32 array of shape `(5,)` with all values in [0, 1].
- **`test_home_pos_returns_copy`** — mutating the returned array doesn't affect subsequent calls.
- **`test_stub_reward_is_zero`** — for each stub task, `compute_reward(...)` returns `0.0`.
- **`test_stub_success_is_false`** — for each stub task, `check_success(...)` returns `False`.
- **`test_stub_terminated_is_false`** — for each stub task, `check_terminated(...)` returns `False`.

### `test_robot_pool.py` — RobotPool

- **`test_acquire_returns_unique_names`** — acquire N robots, all names are distinct.
- **`test_acquire_exhausts_pool`** — acquire all available robots, then next acquire raises (or blocks, depending on design).
- **`test_release_makes_name_available`** — acquire, release, acquire again, get the same name back.
- **`test_thread_safety`** — spawn N threads that each acquire and release, assert no duplicate assignments and no crashes.

### `test_registration.py` — Gymnasium + swm.World integration

- **`test_gym_make_base_env`** — `gym.make("cloudgripper/Gripper-v0")` succeeds and returns a `CloudGripperEnv` with `task=None`.
- **`test_gym_make_task_env`** — `gym.make("cloudgripper/CubePush-v0")` succeeds and returns a `CloudGripperEnv` with `task` set.
- **`test_all_registered_ids_exist`** — all four IDs (`Gripper-v0`, `CubePush-v0`, `CubeStack-v0`, `RopeManip-v0`) are in Gymnasium's registry.
- **`test_swm_world_creates_envpool`** — `swm.World("cloudgripper/Gripper-v0", num_envs=2, image_shape=(64, 64))` creates a World with 2 envs, `world.reset()` succeeds, `world.infos["pixels"]` has shape `(2, 1, 64, 64, 3)`.

## Conventions

- Use `uv` for all dependency management — never raw `pip install`
- All new source goes under `cloudgripper_wm/` package
- Type hints on all public functions
- Tests in `tests/` — use `GripperRobotMock` so tests don't need hardware. See the **Tests** section above for the full test plan and per-file breakdown.
- Configs use Hydra YAML under `cloudgripper_wm/configs/`
- Images are always uint8 numpy arrays, shape (H, W, 3), RGB channel order
- State vectors are float32, normalized to [0, 1]
- Actions are float32 **deltas** in [-max_delta, max_delta] — the env converts to absolute coordinates internally
- The `"pixels"` key in obs/info is **always** the top camera — this is what stable-worldmodel's MegaWrapper picks up for resizing and dataset storage
- **Adding a new task:** (1) subclass `Task` in `cloudgripper_wm/tasks/`, (2) add to `TASK_REGISTRY` in `tasks/__init__.py`, (3) add a `gymnasium.register()` call in `envs/__init__.py`. The env code itself should not change.
- Task reward/success implementations can start as stubs — world model training is self-supervised and doesn't need them. Fill in real logic when RL or MPC evaluation is needed.

## Perception (`cloudgripper_wm/perception/`)

`SceneTracker` finds, names and tracks the objects of cell 20 or 23 with SAM 2.1 in both camera views (`start` at a
parking pose, `step` every control step, `refresh` at later parking poses). On cell 20 each object is prompted by its
colour first, with the shape steps as the fallback. It is written for one setup only: these cells and their
five known objects (two cubes, slot, pan, rope). **Read `cloudgripper_wm/perception/README.md` before changing it**: it
lists every assumption and what is and is not tested. The objects' STL files are in `perception/assets/`.

- Needs the `perception` extra (`sam2`) and a CUDA GPU.
- `scripts/debug/perception_replay.py SWEEP_DIR [...]` (a uv inline-metadata script) replays recorded sweeps through
  it without a robot and writes a side-by-side video.
- `tests/test_perception_inventory.py` checks the naming logic without a GPU.
- Runs inside the env through `FilteredCloudGripperWrapper` (see the collision filter below). `SafeCloudGripperWrapper`
  is the older, separate check that builds a heightmap from a green colour mask.
- In the base view a cube's mask must be one solid square of a cube's size (`cube_like`); otherwise the cube counts as
  not seen from below. This rejects grey robot parts taken for the white cube.
- The robot's reported `x_norm`, `y_norm` trail the real arm by seconds to minutes (measured 2026-11-23): use the
  commanded pose.

## Collision filter (`cloudgripper_wm/collision_filter.py`)

`check(pose0, pose1, states)` decides from `SceneTracker`'s states whether one commanded move may jam the gripper:
pushing a free object is allowed; pushing an object (or a chain of touching objects) against the frame, a finger
against the frame, lowering onto something, and opening the gripper into the frame or into an object the frame stops
are blocked (closing is a grasp and allowed). Pure geometry, no GPU. The module holds cell 23's numbers
by default; `use_cell("robot20")` switches to cell 20's (`CELL20`: glass, tip spacing from the gripper command, tool
offset with a fixed and a turning part, tip heights). Every constant is in the file with how it was measured.

- Tip height against the height command was measured on 2026-11-24 (`Z_TO_TIP_MM`: 6 mm at 0, 15 mm at 0.5, 22 mm at 1):
  the tips never reach the glass, and at the lowest point they are 1 mm below the pan's top and 4 mm below a cube's.
- The finger numbers are for the gripper stuck open (2026-11-24); re-measure after the repair.
- `scripts/debug/perception_replay.py` runs it over a recording that has `cmd_xy` and draws the assumed fingers.
- `tests/test_collision_filter.py` checks the rules on made-up scenes.
- `scripts/debug/calibrate_cam_to_robot.py` makes a cell's pixel-to-robot file (it moves the robot).
- `scripts/debug/teleop_filter.py` (default robot20, records a video of the session) is the by-eye test: each key proposes one move, the live view shows the masks, the
  finger tips now and after the move, and the verdict; `y` sends it, any other key takes it back. It moves the robot.
- A cube that is off the glass is an obstacle too: an object with a cube "stacked" on it grows by one cube all round
  (the cube may lean beside it), and a cube on nothing we see from below gets a square where the top view puts it.
- Held objects (set by a closing move that clamps them, cleared by opening) are no obstacle and make nothing they are
  "stacked on" bigger; when lowering, the space between the tips counts as filled by them, with their bottom
  `HELD_BELOW_MM` (0.6) below the tips'.
- **In the env:** `cloudgripper_wm/envs/filtered_wrapper.py`, `FilteredCloudGripperWrapper(env)` or
  `CloudGripperWorld(robot_names=["robot20"], collision_filter={})`. Reset raises the arm before the env travels
  home, then finds the objects; every step is checked against the objects seen after the previous step and a
  blocked action becomes a zero action (optionally logged to `log_path`). It also narrows the gripper range per cell
  (robot20: 0.1 to 0.825). All robots of one process must be the same cell type (`use_cell` is module-wide).
  `tests/test_filtered_wrapper.py` checks it on the mock robot; `scripts/debug/filtered_env_check.py` is the
  supervised live check (random actions, video). Not yet run on the robot.
- `scripts/debug/teleop_video_speedup.py VIDEO.mp4` shortens a teleop session video: X8 everywhere, X1 around
  blocked moves.

## Play primitives (`cloudgripper_wm/primitives.py`)

`Primitives(env)` on a `FilteredCloudGripperWrapper`: `push`, `reset_objects`, `grasp` / `place` / `pick_place`,
`cube_into_slot`, `rope_drag`. Each is a list of absolute moves, sent through `env.step` as delta actions within the
per-axis step limit `constants.PLAY_MAX_DELTA` (x, y 0.1; height 0.25; rotation 0.25; gripper 0.75), the interface
the world model uses; the collision filter judges every step; a blocked move gives the primitive up (arm to the top height, returns False) and the caller
picks the next one. Written for cell 20; heights, approach distance and waiting times are first guesses.

- `reset_objects` brings objects far from the glass centre back: each rigid object is picked up and put on a free
  spot, or pushed when no grasp fits; the rope is dragged. A pan with its handle towards a frame and its round part
  near it, and a slot with its long side against the low-x or low-y frame, stay where they are (no grasp fits and
  the arm's range, 0.05 to 0.95, ends at the glass edge on the low sides, so nothing gets behind them).
- `grasp` takes a cube by two sides, the slot across its 15 mm width on one of its walls and the pan by its handle (tried 15.5, 13 and 11 mm out from the round part's
  centre). The pan's round part is 20 mm and the open tips leave 22.5 mm: the filter refuses to lower around it.
  When no spot has room for the tips, `_make_room` first pushes the object or a rigid neighbour 25 mm (every candidate
  push is asked of the filter before anything moves), then the grasp is tried once more.
- A grasp closes, eases off by `RECLOSE_EASE` and closes again (a fresh command to a gripper stopped by a wide
  object), then checks in the top view that the object rose with the lift (`LIFT_RISE_PX`); if not, it opens (which
  clears "holding") and gives up. Before lowering what is held (`place`, `cube_into_slot`) `_still_held` checks that
  it moved with the carry in the top view (`DROP_PX`); if it was dropped, "holding" is cleared and nothing is
  lowered. Why: prim_21 (2026-11-26) the slot slipped out on the lift, "holding slot" stayed, the filter ignored the
  slot as held, and the gripper was lowered onto its edge.
- `rope_drag` grasps the rope across it at the first spot, from 12 mm in from the end and then every 8 mm along the
  whole rope, where the filter lets the open tips go down: a rope lying along the frame is taken where it leaves it.
- `cube_out_of_slot(cube, way)`: push (one open tip, 5 mm across the 13.5 mm notch, goes along it at height command
  0.15 and pushes the cube out of the end facing the glass centre; z 0 hit the slot's base, z 0.27 passed over the
  cube; open tips sit 2 to 3 mm above the closed-tip heights in `Z_TO_TIP_MM`), drop, swing or grasp. Slot.stl is stored upside
  down: the notch floor is 5 mm up and a seated cube's top (15 mm) is below the walls (16 mm). Then the
  arm goes home and `look_again()` (the wrapper's refresh) says whether the cube is on the glass: a step alone keeps
  a cube that left the slot "stacked on slot".
- Not written yet: the rope into and out of the slot.
- `tests/test_primitives.py` runs the sequences on the mock robot. `scripts/debug/primitives_check.py --out DIR
  reset push:pan:1,0:30 ...` is the supervised live run (window + video; `--mock` for a dry run).
  `cloudgripper_wm/envs/filter_view.py` is the window/video helper both debug scripts share.
- Run on robot20: `pick_place` of a cube (2026-11-25); `push` of the pan, making room before a grasp, `rope_drag`
  (2026-11-26), `cube_into_slot` and `cube_out_of_slot` by pushing (2026-11-26). Not yet: the pan and slot
  grasps, `reset_objects` as a whole.

## Self-check (`cloudgripper_wm/self_check.py`)

Before play-data collection, every 15 episodes and after any surprise (Kei, 2026-11-27): the arm visits fixed poses
through the filtered env, never below height 0.35 (Kei), and each pose is compared with a reference recorded on the
healthy robot: every axis both ways at the top height around x 0.5, y 0.5, rotation 180 deg; the gripper open,
half-open and closed; then down to 0.7 there and up (tips ~19 mm clear a cube and the slot; at 0.35 nearly any
neighbour blocked the spot). It judges the gripper's yellow block (fixed anchor, does not rotate) and claw in the
top view and the yellow in the base view against the reference with a few px of slack, the block's bottom-edge angle
(tilt, limit 10 deg, Kei: a broken gripper hangs about 30 deg off), the base camera's shift (phase correlation of the
navy frame, two parts with a gap, limit 5 px, Kei; healthy 0.1 to 0.3 px across scenes, blue cube included) and fresh
pictures (the env reuses the last frame when a fetch fails). A failing pose is pictured twice more, 1 s apart, before it
counts: a false alarm stops the collection and is worse than a missed fault (Kei), so limits keep wide margins. The reported pose is printed per pose, not judged
(whether it can be trusted is open: on 2026-11-23 it lagged, cause not established; 2026-11-27 it matched within
0.001); motor currents are not used (Kei: never used).
The reference is `cloudgripper_wm/self_check_ref/<robot>/`, committed to git (Kei: safe from accidental deletion);
`--record` adds the poses it lacks and never replaces one.
`scripts/self_check.py --out DIR [--record] [--ref DIR]` runs it (exit 0 pass, 1 fail, 2 blocked by the scene); DIR
keeps everything for diagnosis (Kei): scene_start.npz (tracker states with masks, `scene_tracker.save_states` /
`load_states`, ~10 KB a scene), raw/ (every picture, retakes included), log.jsonl (numbers per picture), overlays,
report.txt;
`tests/test_self_check.py`. robot20: top-height poses recorded and checked once (2026-11-27); height poses not yet.
