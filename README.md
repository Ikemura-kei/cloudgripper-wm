# World Model Research with [CloudGripper](https://cloudgripper.org/)

## 0. Before You Start

1. **Clone with submodules** — the third-party dependencies won't be present otherwise:
   ```bash
   git clone --recursive https://github.com/Ikemura-kei/cloudgripper-wm.git
   ```

2. **Set `STABLEWM_HOME`** — the stable-worldmodel framework uses this to locate checkpoints and outputs. Make sure it is exported in your shell before running any scripts.

3. **Set `CLOUDGRIPPER_TOKEN`** — required to communicate with the physical robots. This is only needed for data collection; training and offline evaluation do not require it.

## 1. Project Structure

```
cloudgripper-wm/
├── cloudgripper_wm/
│   ├── envs/
│   ├── tasks/
│   ├── policies/
│   └── world.py
├── scripts/
│   ├── data/
│   └── train/
├── tests/
├── data/
├── misc/
└── third_party/
    ├── cloudgripper-api/
    └── stable-worldmodel/
```

| Path | Description |
|------|-------------|
| `cloudgripper_wm/` | Core Python package — environment, tasks, policies, and world wrapper |
| `cloudgripper_wm/envs/` | Gymnasium environment wrapping the CloudGripper HTTP API, plus `RobotPool` for assigning robots to parallel env instances |
| `cloudgripper_wm/tasks/` | Task definitions (reward, success, home position) for cube pushing, stacking, and rope manipulation |
| `cloudgripper_wm/policies/` | Data-collection policies: `StickyRandomPolicy` and `GeometricTrajectoryPolicy` (circle / square / triangle) |
| `cloudgripper_wm/world.py` | `CloudGripperWorld`: thin wrapper around `swm.World` that handles `RobotPool` setup and token resolution |
| `scripts/data/` | Data collection entry point (`collect_cloudgripper.py`), Hydra configs, and inspection/extraction utilities |
| `scripts/train/` | Training entry points (`prejepa.py` for DINO-WM) and their Hydra configs |
| `tests/` | Unit tests — use `GripperRobotMock` so no hardware is needed |
| `data/` | Default output directory for collected datasets (gitignored; a `.placeholder` file keeps the folder tracked) |
| `misc/` | Scratch space for visualizations and debugging outputs (extracted episodes, plots, etc.) — gitignored |
| `third_party/cloudgripper-api/` | HTTP client for the CloudGripper robots *(git submodule)* |
| `third_party/stable-worldmodel/` | World model framework providing training, data pipelines, and planning infrastructure *(git submodule)* |

## 2. Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management. If you don't have it yet:

```bash
pip install uv
```

Then install all dependencies from the lockfile:

```bash
uv sync
```

All scripts should be run via `uv run` to ensure the correct environment is used:

```bash
uv run python <script.py>
```

## 3. Data Collection

Data collection is handled by `scripts/data/collect_cloudgripper.py`. The `episodes` parameter sets the **total target** episode count — re-running with the same config will top up the dataset to that number without duplicating existing episodes.

Example usages (see the top of the script for the full list):

```bash
# Default config (random policy)
uv run python scripts/data/collect_cloudgripper.py output=data/my_data

# Geometric trajectory policy
uv run python scripts/data/collect_cloudgripper.py --config-name cloudgripper_geometric output=data/my_data
```

> **Note:** Hydra uses `--config-name` (with a dash), not `--config_name`.

## 4. Training World Models

Currently supported world models:

- **LeWM** — `scripts/train/lewm.py`

Checkpoints are saved to `$STABLEWM_HOME/checkpoints/<model_name>/<datetime_xxx>`, where `xxx` is a randomly chosen three-letter suffix for uniqueness. The same name appears as the run name on the WandB dashboard.

## 5. Evaluating World Models
