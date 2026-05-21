# CLAUDE.md — cloudgripper-wm

## Project Overview

This project trains a **world model** on the [CloudGripper](https://cloudgripper.org/) real cloud robotics facility at KTH Stockholm. CloudGripper is an open-source testbed with **32 small robot arm cells** controllable over HTTP APIs. The world model framework is [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) (`stable_worldmodel`), included as an editable git submodule under `third_party/stable-worldmodel`.

The goal is to collect real-robot interaction data in parallel across multiple robots, store it in a format compatible with `stable-worldmodel`, and train world models (JEPA-family: DINO-WM, LeWM, etc.) on that data.

## Repository Structure

```
cloudgripper-wm/
├── cloudgripper_wm/          # Main package (to be created)
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py        # gymnasium.register() for all task env IDs
│   │   ├── cloudgripper_env.py  # Core Gymnasium env wrapping CloudGripper API
│   │   └── robot_pool.py     # Thread-safe pool assigning physical robots to env instances
│   ├── tasks/
│   │   ├── __init__.py        # TASK_REGISTRY dict, get_task() factory
│   │   ├── base.py            # Abstract Task base class
│   │   ├── cube_push.py       # CubePushTask
│   │   ├── cube_stack.py      # CubeStackTask
│   │   └── rope_manip.py      # RopeManipTask
│   ├── policies/
│   │   ├── __init__.py
│   │   ├── random_push.py    # Scripted random pushing policy
│   │   └── heuristic_grasp.py  # Scripted pick-and-place policy
│   └── configs/               # Hydra YAML configs for collection and training
│       ├── collect.yaml
│       └── train.yaml
├── scripts/
│   ├── collect.py             # Data collection entry point
│   └── train.py               # Training entry point
├── tests/
│   ├── conftest.py            # Shared fixtures (mock robot, env factories)
│   ├── test_env_base.py       # Task-agnostic CloudGripperEnv tests
│   ├── test_env_tasks.py      # Task-specific env tests (one per task)
│   ├── test_tasks.py          # Task classes in isolation (reward, success, termination)
│   ├── test_robot_pool.py     # RobotPool thread-safety tests
│   └── test_registration.py   # Gymnasium registration + swm.World integration
├── third_party/
│   └── stable-worldmodel/     # Git submodule (editable dep)
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

### Dev robot

- **robot23** is the designated development robot for testing without hardware setup
- Set the token before any real-robot code: `export CLOUDGRIPPER_TOKEN=<token>`
  - Do NOT commit the token to any file — keep it in your shell environment only
- Quick smoke test on real hardware: `uv run python scripts/test_real_robot.py`
  - Defaults to robot23, 10 random steps, 0.5 s dwell
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
- **`"pixels"`**: top camera image (primary — this is what MegaWrapper/World uses)
- **`"pixels_base"`**: base camera image (secondary, stored in info for dataset but not required by MegaWrapper)
- **`"state"`**: flat float32 vector of `[x, y, z, rotation_normalized, gripper]`, all in [0, 1]. This is the *commanded target position* (`self._target_pos`), not the raw readback from the robot (which may lag).

### Action space (delta)
- `Box(-max_delta, max_delta, shape=(5,), dtype=float32)` → `[Δx, Δy, Δz, Δrotation, Δgripper]`
- **Delta actions, not absolute.** The env maintains an internal `self._target_pos: np.ndarray` of shape `(5,)` representing `[x, y, z, rotation_norm, gripper]`, all in [0, 1].
- Each step: `self._target_pos = np.clip(self._target_pos + action, 0.0, 1.0)`, then the clipped absolute values are sent to the robot API.
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
- On `reset()`, set `self._target_pos` to `self.task.home_pos()` if task is set, otherwise `DEFAULT_HOME_POS` `[0.5, 0.5, 1.0, 0.0, 1.0]` (center xy, top z, 0° rotation, gripper open).
- The `"state"` in the observation space reports `self._target_pos` (the intended target). Optionally, a `"state_actual"` key can store the readback from `robot.get_state()` if the caller wants to compare.
- Do NOT re-sync `_target_pos` from `get_state()` every step — it adds latency and the robot may not have reached the target yet. Only re-sync on `reset()` if needed.

### Step timing
- Real robot has HTTP latency + physical movement time
- After sending commands, sleep for a configurable `dwell_time` (default ~0.5s) before reading observations
- This means data collection is slow compared to simulation — parallel robots are essential
- The `dwell_time` should be long enough that the robot approximately reaches `_target_pos` before the next observation is captured. With small `max_delta` values this is naturally satisfied.

### Reset behavior
- Move to home position: `self.task.home_pos()` if task is set, otherwise `DEFAULT_HOME_POS` `[0.5, 0.5, 1.0, 0.0, 1.0]` (center xy, top z, 0° rotation, gripper open)
- Set `self._target_pos` accordingly
- Send absolute commands to robot: `move_xy`, `move_z`, `rotate`, `gripper_open`
- Object rearrangement on the workspace is manual/out-of-scope for now
- Episodes are fixed-length (truncation via `max_episode_steps`). Early termination only possible when a task is set and `task.check_terminated()` returns True.

### Multi-robot parallelism via RobotPool
- `World` creates `num_envs` copies of the env via `gym.make` with identical kwargs
- A thread-safe `RobotPool` singleton assigns a unique robot name to each env instance on `__init__`
- `env.close()` returns the robot name to the pool
- Constraint: `num_envs ≤ number_of_available_robots`

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

DEFAULT_HOME_POS = np.array([0.5, 0.5, 1.0, 0.0, 1.0], dtype=np.float32)

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

## Implementation Order

1. **`Task` base class + stubs** — `base.py` with abstract interface + `DEFAULT_HOME_POS`, then `CubePushTask`, `CubeStackTask`, `RopeManipTask` as stubs (reward=0, success=False). Task registry in `tasks/__init__.py` with `get_task()` returning `None` for `task_name=None`. **Tests:** `test_tasks.py`.
2. **`CloudGripperEnv`** — single robot, Gymnasium-compliant, `task: str | None = None`. When `task=None`: reward=0, terminated=False, no success key. When task is set: delegate to Task object. Include mock support via `GripperRobotMock`. **Tests:** `test_env_base.py`, `test_env_tasks.py`.
3. **`RobotPool`** — thread-safe robot name assignment for multi-env. **Tests:** `test_robot_pool.py`.
4. **Gymnasium registration** — `cloudgripper/Gripper-v0` (task-agnostic) + one ID per task. **Tests:** `test_registration.py`.
5. **Task-agnostic data collection** — `world.collect()` with `cloudgripper/Gripper-v0` + `RandomPolicy`. This is the primary path for world model training data.
6. **Scripted policies** — task-aware pushing/grasping policies for richer training data.
7. **Training** — adapt DINO-WM baseline from `stable-worldmodel/scripts/train/prejepa.py` to CloudGripper observations.
8. **Reward & success implementations** — fill in real `compute_reward()` / `check_success()` for each task when running RL baselines or MPC evaluation.

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