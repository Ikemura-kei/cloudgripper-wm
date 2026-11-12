"""Train LeWM on a flat 2D ball dataset.

Reuses lewm.py's training loop as-is (dataset-agnostic) with a flat2d-specific
Hydra config (scripts/train/config/lewm_flat2d_ball.yaml -> data/flat2d_ball.yaml).

`dataset_name` selects which collected dataset to train on, and is the only
thing that should differ between runs meant to be compared — `seed` is fixed
in the config, so two runs share an initialisation and a shuffle order and the
data is the single variable. Checkpoints carry the dataset name, so the runs
cannot overwrite each other.

Usage:
    uv run python scripts/train/lewm_flat2d_ball.py
    uv run python scripts/train/lewm_flat2d_ball.py dataset_name=flat2d_ball_green
    uv run python scripts/train/lewm_flat2d_ball.py trainer.max_epochs=200
"""

import hydra

from lewm import train


@hydra.main(version_base=None, config_path='./config', config_name='lewm_flat2d_ball')
def run(cfg):
    train(cfg)


if __name__ == '__main__':
    run()
