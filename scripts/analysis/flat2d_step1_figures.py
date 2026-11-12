"""Figures for Step 1 (colour sensitivity of a single model).

Reads what flat2d_step1_colour_sensitivity.py saved plus the embedding cache,
and renders four figures:

  fig1_similarity.png — how far the representation moves when the ball is
                        recoloured, against the spread it occupies anyway
  fig2_collapse.png   — PCA of the representation under each input colour, and
                        the spread / effective-dimension collapse that goes
                        with it
  fig3_decoded.png    — the decoded latent: ball position read out of the
                        representation, red input vs green input
  fig4_frames.png     — that readout drawn back onto the real frames

Identity is carried by colour *and* marker shape *and* a direct label; ball
position, being a magnitude, uses a single-hue sequential ramp. See
flat2d_style.py for the CVD measurements behind the categorical pair.
"""

import io
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flat2d_style as S  # noqa: E402
from flat2d_common import REPO, HISTORY  # noqa: E402

S.use_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = REPO / 'outputs' / 'flat2d_step1'
CACHE = REPO / 'outputs' / 'flat2d_cache'

# dataviz sequential ramp: single hue, light -> dark (blue 150 -> 700)
SEQ = LinearSegmentedColormap.from_list(
    'blue_seq', ['#b7d3f6', '#6da7ec', '#2a78d6', '#184f95', '#0d366b'])


def load():
    z = np.load(OUT / 'step1_arrays.npz')
    res = json.load(open(OUT / 'step1_results.json'))
    red = np.load(CACHE / 'mred1_val_red.npz')
    grn = np.load(CACHE / 'mred1_val_green.npz')
    return z, res, red, grn


def fig_similarity(z, res):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6))
    for col, rep in enumerate(('cls', 'proj')):
        r = res[rep]
        ax = axes[0, col]
        pair, rand = z[f'{rep}|cos_pair'], z[f'{rep}|cos_rand']
        bins = np.linspace(min(pair.min(), rand.min()), 1.0, 60)
        ax.hist(rand, bins=bins, color=S.NEUTRAL, alpha=0.6, label='different scene, same colour')
        ax.hist(pair, bins=bins, color=S.RED, alpha=0.85, label='same scene, recoloured')
        ax.set_title(f'{rep} — cosine similarity', loc='left')
        ax.set_xlabel('cosine similarity'); ax.set_ylabel('windows')
        ax.legend(loc='upper left')
        ax.annotate(f'paired {r["cos_paired_mean"]:.4f}\nrandom {r["cos_random_mean"]:.4f}',
                    xy=(0.97, 0.52), xycoords='axes fraction', ha='right',
                    fontsize=8, color=S.INK, fontweight='bold')
        S.despine(ax)

        ax = axes[1, col]
        dc, dsx = z[f'{rep}|d_colour'], z[f'{rep}|d_scene']
        bins = np.linspace(0, max(dc.max(), dsx.max()), 60)
        ax.hist(dsx, bins=bins, color=S.NEUTRAL, alpha=0.6, label='between different scenes')
        ax.hist(dc, bins=bins, color=S.RED, alpha=0.85, label='red vs green, same scene')
        ax.set_title(f'{rep} — distance moved', loc='left')
        ax.set_xlabel('L2 distance'); ax.set_ylabel('windows')
        ax.legend(loc='upper right')
        ax.annotate(f'ratio {r["ratio"]:.3f}', xy=(0.97, 0.52), xycoords='axes fraction',
                    ha='right', fontsize=8, color=S.INK, fontweight='bold')
        S.despine(ax)
    fig.suptitle('Step 1a — recolouring the ball moves the representation as far as '
                 'changing the scene entirely',
                 fontsize=11, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / 'fig1_similarity.png'); plt.close(fig)


def fig_collapse(red, grn):
    x = red['x']
    fig = plt.figure(figsize=(10.5, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05])

    # PCA fitted on the RED embeddings; green projected into the same basis,
    # so the panels share axes and the collapse is directly comparable.
    rep = 'proj'
    a, b = red[rep][:, HISTORY], grn[rep][:, HISTORY]
    mu = a.mean(0)
    U, Sv, Vt = np.linalg.svd(a - mu, full_matrices=False)
    pa, pb = (a - mu) @ Vt[:2].T, (b - mu) @ Vt[:2].T
    lim = np.abs(pa).max() * 1.15

    for i, (tag, p) in enumerate((('red', pa), ('green', pb))):
        ax = fig.add_subplot(gs[0, i])
        sc = ax.scatter(p[:, 0], p[:, 1], c=x, cmap=SEQ, s=7, linewidths=0)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('PC1 (fitted on red)'); ax.set_ylabel('PC2' if i == 0 else '')
        ax.set_title(f'{tag} input', loc='left', color=S.COND[tag])
        S.despine(ax)
        if i == 1:
            ax.annotate('collapsed', xy=(0, 0), xytext=(0.5 * lim, 0.5 * lim),
                        arrowprops=dict(arrowstyle='->', color=S.INK, lw=1.2),
                        fontsize=9, fontweight='bold', color=S.INK)
        if i == 0:
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label('true ball x', fontsize=8); cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    labels, vals = [], []
    for r in ('cls', 'proj'):
        aa, bb = red[r][:, HISTORY], grn[r][:, HISTORY]
        labels.append(r)
        vals.append(bb.std(0).mean() / aa.std(0).mean())
    bars = ax.bar(range(len(vals)), vals, color=[S.GREEN] * len(vals), width=0.5)
    for bb_, v in zip(bars, vals):
        ax.text(bb_.get_x() + bb_.get_width() / 2, v, f'{v:.3f}', ha='center',
                va='bottom', fontsize=8.5, color=S.INK, fontweight='bold')
    ax.axhline(1.0, color=S.AXIS, lw=1.0, ls='--')
    ax.text(0.98, 1.0, 'red-input spread', transform=ax.get_yaxis_transform(),
            ha='right', va='bottom', fontsize=7.5, color=S.INK2)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels)
    ax.set_ylabel('green spread / red spread')
    ax.set_ylim(0, 1.15)
    ax.set_title('representational spread', loc='left')
    S.despine(ax)

    fig.suptitle('Step 1b — on an unseen ball colour the representation collapses: '
                 'different scenes map to nearly one point',
                 fontsize=11, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / 'fig2_collapse.png'); plt.close(fig)


def fig_decoded(z, res):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    true = z['cls|true']

    ax = axes[0]
    ax.scatter(true, z['cls|pred_red'], s=6, alpha=0.4, color=S.RED,
               marker=S.MARK['red'], linewidths=0, label='red input')
    lim = [true.min() - 0.05, true.max() + 0.05]
    ax.plot(lim, lim, color=S.AXIS, lw=1.0, ls='--', zorder=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('true ball x (image widths)'); ax.set_ylabel('decoded ball x')
    ax.set_title('cls — red input (in distribution)', loc='left')
    ax.annotate(f'R² {res["cls"]["probe_r2_red"]:.3f}\nMAE {res["cls"]["probe_mae_red"]:.4f}',
                xy=(0.04, 0.80), xycoords='axes fraction', fontsize=8,
                color=S.INK, fontweight='bold')
    ax.legend(loc='lower right'); S.despine(ax)

    ax = axes[1]
    pg = z['cls|pred_green']
    ax.scatter(true, pg, s=6, alpha=0.4, color=S.GREEN, marker=S.MARK['green'],
               linewidths=0, label='green input')
    ax.plot(lim, lim, color=S.AXIS, lw=1.0, ls='--', zorder=0)
    ax.set_xlim(lim)
    ax.set_ylim(min(pg.min() - 0.1, lim[0]), lim[1])
    ax.set_xlabel('true ball x (image widths)'); ax.set_ylabel('decoded ball x')
    ax.set_title('cls — green input (unseen colour)', loc='left')
    ax.annotate(f'R² {res["cls"]["probe_r2_green"]:.1f}\n'
                f'decoded span {pg.max()-pg.min():.2f} vs true {true.max()-true.min():.2f}',
                xy=(0.04, 0.80), xycoords='axes fraction', fontsize=8,
                color=S.INK, fontweight='bold')
    ax.legend(loc='lower right'); S.despine(ax)

    ax = axes[2]
    labels, vals, cols = [], [], []
    for rep in ('cls', 'proj'):
        for tag in ('red', 'green'):
            labels.append(f'{rep}\n{tag}'); vals.append(res[rep][f'probe_mae_{tag}'])
            cols.append(S.COND[tag])
    bars = ax.bar(range(len(vals)), vals, color=cols, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.3f}', ha='center', va='bottom',
                fontsize=7.5, color=S.INK, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel('MAE (image widths, log scale)')
    ax.set_title('probe fitted on red frames', loc='left')
    S.despine(ax)

    fig.suptitle('Step 1c — the position probe reads the red input accurately and the '
                 'green input not at all',
                 fontsize=11, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / 'fig3_decoded.png'); plt.close(fig)


def fig_frames(z, n=4):
    import lance
    import stable_worldmodel as swm
    val_idx, true = z['val_idx'], z['cls|true']
    pr_r, pr_g = z['cls|pred_red'], z['cls|pred_green']
    tabs = {t: lance.dataset(str(REPO / '.stable_worldmodel' / 'datasets' /
                                 f'flat2d_ball_{t}.lance')).to_table(
        columns=['episode_idx', 'step_idx', 'pixels']).to_pandas() for t in ('red', 'green')}
    ds = swm.data.load_dataset(
        str(REPO / '.stable_worldmodel' / 'datasets' / 'flat2d_ball_red.lance'),
        transform=None, num_steps=HISTORY + 1, frameskip=1, keys_to_load=['pixels'])
    clips = ds.clip_indices

    pick = np.linspace(0, len(val_idx) - 1, n).astype(int)
    fig, axes = plt.subplots(2, n, figsize=(2.7 * n, 5.6))
    for col, p in enumerate(pick):
        ep, start = clips[int(val_idx[p])]
        step = start + HISTORY
        for row, tag in enumerate(('red', 'green')):
            df = tabs[tag]
            m = (df.episode_idx == ep) & (df.step_idx == step)
            img = np.array(Image.open(io.BytesIO(df.loc[m, 'pixels'].iloc[0])).convert('RGB'))
            Hh, W = img.shape[:2]
            ax = axes[row, col]
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.axvline(true[p] * W, color='#0b0b0b', lw=1.6, ls='--')
            dec = (pr_r if tag == 'red' else pr_g)[p]
            if 0 <= dec <= 1:
                ax.axvline(dec * W, color=S.COND[tag], lw=2.2)
                note = f'decoded err {abs(dec - true[p]) * W:.1f} px'
            else:
                ax.annotate(f'decoded x = {dec:+.2f}\n(off image)', xy=(0.04, 0.10),
                            xycoords='axes fraction', fontsize=8, fontweight='bold',
                            color=S.GREEN)
                note = 'decode off image'
            ax.set_title(note, loc='left', fontsize=8, color=S.INK2)
            if col == 0:
                ax.set_ylabel(f'{tag} input', fontsize=9.5, color=S.COND[tag],
                              fontweight='bold')
    fig.suptitle('Step 1d — decoded ball position (solid) vs ground truth (dashed). '
                 'One model; only the ball colour differs.',
                 fontsize=11, fontweight='bold', color=S.INK, x=0.01, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / 'fig4_frames.png'); plt.close(fig)


if __name__ == '__main__':
    z, res, red, grn = load()
    fig_similarity(z, res)
    fig_collapse(red, grn)
    fig_decoded(z, res)
    fig_frames(z)
    print('figures ->', OUT)
