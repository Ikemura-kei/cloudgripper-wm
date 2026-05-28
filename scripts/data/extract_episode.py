"""Extract a random (or specified) episode from a Lance dataset.

Saves:
  misc/episode_{idx:04d}_{dataset_name}/
    step_0000_top.png        top camera image per step
    step_0000_base.png       base camera image per step (if available)
    actions.csv              one row per step, columns: dx dy dz drot dgrip
    states.csv               one row per step, columns: x y z rot grip

Usage:
    uv run python scripts/data/extract_episode.py data/cloudgripper.lance
    uv run python scripts/data/extract_episode.py data/cloudgripper.lance --episode 42
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
import lance
import numpy as np


ACTION_COLS = ['dx', 'dy', 'dz', 'drot', 'dgrip']
STATE_COLS  = ['x',  'y',  'z',  'rot',  'grip']


def decode_jpeg(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='Path to .lance dataset')
    parser.add_argument('--episode', type=int, default=None,
                        help='Episode index to extract (default: random)')
    args = parser.parse_args()

    ds = lance.dataset(args.dataset)
    episodes = sorted(set(
        ds.to_table(columns=['episode_idx']).column('episode_idx').to_pylist()
    ))
    if not episodes:
        raise RuntimeError('Dataset contains no episodes.')

    ep_idx = args.episode if args.episode is not None else random.choice(episodes)
    if ep_idx not in episodes:
        raise ValueError(f'Episode {ep_idx} not found. Available: {episodes[0]}–{episodes[-1]}')

    dataset_name = Path(args.dataset).stem
    out_dir = Path('misc') / f'episode_{ep_idx:04d}_{dataset_name}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Extracting episode {ep_idx} → {out_dir}/')

    table  = ds.to_table(filter=f'episode_idx = {ep_idx}').to_pydict()
    n      = len(table['step_idx'])
    order  = sorted(range(n), key=lambda i: table['step_idx'][i])
    has_base  = 'pixels_base' in table
    has_state = 'state' in table

    with open(out_dir / 'actions.csv', 'w', newline='') as af, \
         open(out_dir / 'states.csv',  'w', newline='') as sf:

        aw = csv.writer(af)
        sw = csv.writer(sf)
        aw.writerow(['step'] + ACTION_COLS)
        sw.writerow(['step'] + STATE_COLS)

        for rank, i in enumerate(order):
            step = table['step_idx'][i]

            # images
            top = decode_jpeg(table['pixels'][i])
            cv2.imwrite(str(out_dir / f'step_{rank:04d}_top.png'), top)

            if has_base:
                base = decode_jpeg(table['pixels_base'][i])
                cv2.imwrite(str(out_dir / f'step_{rank:04d}_base.png'), base)

            # action
            action = list(table['action'][i])
            aw.writerow([step] + [f'{v:.6f}' for v in action])

            # state
            if has_state:
                state = list(table['state'][i])
                sw.writerow([step] + [f'{v:.6f}' for v in state])

    if not has_state:
        (out_dir / 'states.csv').unlink()
        print('  (no state column found — states.csv not written)')

    n_img = len(order)
    print(f'  {n_img} steps saved  |  {"top+base" if has_base else "top only"}  |  actions.csv{"  states.csv" if has_state else ""}')


if __name__ == '__main__':
    main()
