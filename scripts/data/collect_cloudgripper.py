"""Collect data from CloudGripper robots using a configured policy.

`output` is the **folder** that will contain `{name}.lance` and `config.yaml`.
`episodes` is the **total target** episode count.  Re-run with the same config
to top up; only (target - existing) more episodes are collected.

Usage:
    uv run python scripts/data/collect_cloudgripper.py
    uv run python scripts/data/collect_cloudgripper.py robots=[robot1,robot2] episodes=100
    uv run python scripts/data/collect_cloudgripper.py use_mock=true output=data/mock
    uv run python scripts/data/collect_cloudgripper.py --config-name cloudgripper_geometric
"""

from pathlib import Path

import hydra
import lance
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from cloudgripper_wm.world import CloudGripperWorld


def _lance_path(output: str) -> str:
    """Derive the Lance dataset path inside the output folder."""
    p = Path(output)
    return str(p / (p.name + '.lance'))


def _config_path(output: str) -> Path:
    return Path(output) / 'config.yaml'


def _count_existing_episodes(output: str) -> int:
    """Return number of episodes already in the Lance dataset, or 0 if absent."""
    lp = _lance_path(output)
    if not Path(lp).exists():
        return 0
    try:
        col = lance.dataset(lp).to_table(columns=["episode_idx"]).column("episode_idx").to_pylist()
        return max(col) + 1 if col else 0
    except Exception:
        return 0


def _save_config(cfg: DictConfig, output: str) -> None:
    config_path = _config_path(output)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, config_path)
    logging.info(f'Config saved → {config_path}')


def _check_config_compatibility(cfg: DictConfig, output: str) -> None:
    """Raise if a saved config exists and anything other than `episodes` changed."""
    config_path = _config_path(output)
    if not config_path.exists():
        return
    saved = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    current = OmegaConf.to_container(cfg, resolve=False)
    saved.pop('episodes', None)
    current.pop('episodes', None)
    if current != saved:
        all_keys = set(saved) | set(current)
        changed = [k for k in all_keys if saved.get(k) != current.get(k)]
        raise ValueError(
            f'Config mismatch on resume — changed keys: {changed}. '
            f'Use a different output path or delete the existing dataset to start fresh.'
        )


@hydra.main(version_base=None, config_path='./config', config_name='cloudgripper')
def run(cfg: DictConfig) -> None:
    lance_out = _lance_path(cfg.output)

    n_existing = _count_existing_episodes(cfg.output)
    if n_existing > 0:
        _check_config_compatibility(cfg, cfg.output)
    to_collect = max(0, cfg.episodes - n_existing)

    if n_existing > 0:
        logging.info(
            f'Dataset exists: {n_existing} episodes. '
            f'Target: {cfg.episodes}. Collecting {to_collect} more.'
        )

    if to_collect == 0:
        logging.info('Target episode count already reached, nothing to collect.')
        return

    seed_start = cfg.seed + n_existing
    _save_config(cfg, cfg.output)

    world = CloudGripperWorld(
        robot_names=list(cfg.robots),
        use_mock=cfg.use_mock,
        **cfg.world,
    )

    world.set_policy(instantiate(cfg.policy, seed=seed_start))

    max_retries = 1000
    try:
        for ep in range(to_collect):
            seed = seed_start + ep
            for attempt in range(1, max_retries + 1):
                try:
                    world.collect(lance_out, episodes=1, seed=seed)
                    break
                except Exception as exc:
                    logging.warning(f'Episode {n_existing + ep + 1} attempt {attempt}/{max_retries} failed: {exc}')
                    if attempt == max_retries:
                        raise RuntimeError(
                            f'Episode {n_existing + ep + 1} failed after {max_retries} attempts'
                        ) from exc
                    seed += to_collect  # offset so retries don't collide with primary episode seeds
            logging.info(f'Episode {n_existing + ep + 1}/{cfg.episodes} saved → {lance_out}')
    except KeyboardInterrupt:
        logging.warning('Interrupted — closing world and releasing robots.')
    finally:
        world.close()
    logging.success(f'Collected {to_collect} episodes → {lance_out} (total: {n_existing + to_collect})')


if __name__ == '__main__':
    run()
