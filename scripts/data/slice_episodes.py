"""Create a new Lance dataset containing only episodes up to a given index.

Usage:
    uv run python scripts/data/slice_episodes.py \
        data/first_medium_sized \
        data/first_medium_sized_ep0-645 \
        --max-episode 645
"""

import argparse
from pathlib import Path

import lance
import pyarrow as pa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('src',  help='Source .lance dataset path')
    parser.add_argument('dst',  help='Destination .lance dataset path')
    parser.add_argument('--max-episode', type=int, required=True,
                        help='Keep episodes with episode_idx <= this value (inclusive)')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if dst.exists():
        raise FileExistsError(f'Destination already exists: {dst}')

    ds = lance.dataset(str(src))
    total_before = ds.count_rows()

    table = ds.to_table(filter=f'episode_idx <= {args.max_episode}')
    total_after = len(table)

    episodes_kept = len(set(table['episode_idx'].to_pylist()))
    print(f'Source rows  : {total_before}')
    print(f'Kept rows    : {total_after}  (episodes 0 – {args.max_episode}, {episodes_kept} episodes)')
    print(f'Writing → {dst} …')

    lance.write_dataset(table, str(dst))
    print('Done.')


if __name__ == '__main__':
    main()
