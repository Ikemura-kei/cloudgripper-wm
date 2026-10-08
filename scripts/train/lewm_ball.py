"""Train LeWM on the CloudGripper MuJoCo ball dataset.

Reuses lewm.py's training loop as-is (dataset-agnostic) with a ball-specific
Hydra config (scripts/train/config/lewm_ball.yaml -> data/cloudgripper_ball.yaml).

Usage:
    uv run python scripts/train/lewm_ball.py
    uv run python scripts/train/lewm_ball.py trainer.max_epochs=200
"""

import hydra

from lewm import train


@hydra.main(version_base=None, config_path='./config', config_name='lewm_ball')
def run(cfg):
    train(cfg)


if __name__ == '__main__':
    run()
