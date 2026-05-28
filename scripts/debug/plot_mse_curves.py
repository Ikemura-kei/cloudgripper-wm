"""Plot multiple MSE curves from eval_mse_curve output directories.

Each subdirectory under the given path is expected to contain:
  mse_data.npz  — output of eval_mse_curve.py
  config.yaml   — Hydra config (used to extract a legend label)

The legend label is inferred from the checkpoint filename:
  weights_epoch_50.pt  →  epoch 50
  weights.pt           →  the filename stem

Usage:
    uv run python scripts/debug/plot_mse_curves.py /path/to/mse_eval/run_group
    uv run python scripts/debug/plot_mse_curves.py /path/to/mse_eval/run_group --out comparison.png
    uv run python scripts/debug/plot_mse_curves.py /path/to/mse_eval/run_group --no-std
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _legend_label(config_path: Path) -> str:
    text = config_path.read_text()
    # extract checkpoint: line
    m = re.search(r'checkpoint:\s*(\S+)', text)
    if not m:
        return config_path.parent.name
    ckpt = Path(m.group(1)).stem          # e.g. "weights_epoch_50"
    epoch_m = re.search(r'epoch[_\s]?(\d+)', ckpt, re.IGNORECASE)
    return f'epoch {epoch_m.group(1)}' if epoch_m else ckpt


def _sort_key(label: str) -> int:
    m = re.search(r'\d+', label)
    return int(m.group()) if m else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Directory containing timestamped eval subdirs')
    parser.add_argument('--out', default=None,
                        help='Output path for the plot (default: <dir>/comparison.png)')
    parser.add_argument('--no-std', action='store_true',
                        help='Suppress ±1 std shading')
    args = parser.parse_args()

    base = Path(args.dir)
    runs = sorted(
        [d for d in base.iterdir() if d.is_dir() and (d / 'mse_data.npz').exists()]
    )
    if not runs:
        raise FileNotFoundError(f'No eval subdirs with mse_data.npz found in {base}')

    # collect and sort by epoch
    entries = []
    for run_dir in runs:
        data  = np.load(run_dir / 'mse_data.npz')
        label = _legend_label(run_dir / 'config.yaml')
        entries.append((label, data))
    entries.sort(key=lambda x: _sort_key(x[0]))

    # ---- plot ---------------------------------------------------------- #
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if len(entries) > len(colors):
        colors = [plt.cm.tab20(i / len(entries)) for i in range(len(entries))]

    fig, ax = plt.subplots(figsize=(10, 5))

    for (label, data), color in zip(entries, colors):
        steps    = data['steps']
        mean_mse = data['mean']
        std_mse  = data['std']
        ax.plot(steps, mean_mse, marker='o', markersize=3,
                label=label, color=color)
        if not args.no_std:
            ax.fill_between(steps,
                            mean_mse - std_mse,
                            mean_mse + std_mse,
                            alpha=0.08, color=color)
            ax.plot(steps, mean_mse - std_mse, linestyle='--',
                    linewidth=0.8, color=color, alpha=0.6)
            ax.plot(steps, mean_mse + std_mse, linestyle='--',
                    linewidth=0.8, color=color, alpha=0.6)

    ax.set_xlabel('Prediction step')
    ax.set_ylabel('Embedding MSE')
    ax.set_title(f'Autoregressive MSE — {base.name}')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = Path(args.out) if args.out else base / 'comparison.png'
    plt.savefig(out, dpi=150)
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
