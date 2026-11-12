"""Merge fixed-colour datasets into one, with the colours exactly balanced.

Why not just collect one dataset with the colour sampled per episode: sampling
is random, so the counts come out uneven. Measured on a 2000-episode draw from
three colours: red 631, green 673, blue 696 — a 10% skew, with the training
colour the *least* represented. For a study about how an encoder treats
colours, that is a confound that can masquerade as a finding.

So each colour is collected separately at a known episode count, and this
merges them. Episode ids are reassigned round-robin (red, green, blue, red, …)
so the colours are interleaved rather than lying in three contiguous blocks —
which keeps any episode-ordered operation from seeing one colour at a time.

Exact balance in the train/val split is a separate matter, handled by
`stratify_by` in the training config.

Usage:
    uv run python scripts/data/merge_colour_parts.py \\
        --parts _rgbpart_red _rgbpart_green _rgbpart_blue \\
        --out rolling_ball_rgb_default
"""

import argparse
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa

DATA = Path(__file__).resolve().parents[2] / '.stable_worldmodel' / 'datasets'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    tabs, order = [], []
    for part in args.parts:
        t = lance.dataset(str(DATA / f'{part}.lance')).to_table()
        eps = sorted(set(t.column('episode_idx').to_pylist()))
        tabs.append((part, t, eps))
        order.append(eps)
        print(f'{part:<18}{t.num_rows:>8} rows  {len(eps):>5} episodes')

    # round-robin the episode ids so colours interleave
    remap, nxt = {}, 0
    pools = [list(e) for e in order]
    while any(pools):
        for i, pool in enumerate(pools):
            if pool:
                remap[(i, pool.pop(0))] = nxt
                nxt += 1
    print(f'\nreassigned {nxt} episode ids, interleaved across {len(args.parts)} parts')

    out = []
    for i, (part, t, _) in enumerate(tabs):
        old = np.asarray(t.column('episode_idx').to_pylist())
        new = np.array([remap[(i, int(e))] for e in old], dtype=np.int32)
        cols = {name: t.column(name) for name in t.schema.names}
        cols['episode_idx'] = pa.array(new, type=pa.int32())
        out.append(pa.table(cols, schema=t.schema))

    merged = pa.concat_tables(out)
    # sort by (episode, step) so the merged file reads in episode order
    idx = np.lexsort((np.asarray(merged.column('step_idx').to_pylist()),
                      np.asarray(merged.column('episode_idx').to_pylist())))
    merged = merged.take(pa.array(idx))

    path = DATA / f'{args.out}.lance'
    if path.exists():
        raise SystemExit(f'{path} already exists — remove it first')
    lance.write_dataset(merged, str(path))
    print(f'wrote {path}  ({merged.num_rows} rows)')


if __name__ == '__main__':
    main()
