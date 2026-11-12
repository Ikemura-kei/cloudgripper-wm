"""Shared matplotlib style for the flat2d red/green study figures.

Colour choices are computed, not eyeballed. The two conditions are encoded with
the semantically correct red and green, which is a deuteranopia-risky pair, so
it was measured with scripts/analysis/cvd_check.py before being adopted:

    pair                    normal    deutan   protan
    semantic red/green        33.1      10.5      9.0     PASS (>=15 / >=8)
    palette blue/orange       33.6      31.6     24.5     PASS

Red/green clears both gates but with far less margin than an abstract pair, so
every figure pairs colour with a second channel — distinct marker shapes, and
direct labels on the marks — and never relies on hue alone to carry identity.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# dataviz reference palette, light surface
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
RULE = '#c3c2b7'

RED = '#e34948'     # categorical slot 8
GREEN = '#008300'   # categorical slot 6
NEUTRAL = '#898781'

COND = {'red': RED, 'green': GREEN}
MARK = {'red': 'o', 'green': 's'}   # second channel: shape, never hue alone


def use_style() -> None:
    plt.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'axes.labelsize': 9,
        'axes.labelcolor': INK2,
        'axes.edgecolor': AXIS,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.frameon': False,
        'legend.fontsize': 8,
        'lines.linewidth': 2.0,
        'lines.markersize': 4,
        'figure.dpi': 150,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
    })


def despine(ax, keep=('left', 'bottom')) -> None:
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(side in keep)
