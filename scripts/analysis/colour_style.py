"""Figure style for the colour-generalisation study.

Reuses the base style from the red/green study. The one addition is deliberate
restraint: the five ways of handling a recoloured embedding are *not* a
categorical palette. A five-hue set could not clear the CVD gate — the best
search result left adjacent pairs at dE 4 under deuteranopia, well under the
floor of 8 — and a five-step lightness ramp fared no better. So the figures
facet by method instead of colouring by it: one panel per method, titled, each
drawn in the single accent below. Identity is carried by the panel title, which
no colour vision deficiency can affect.

The two hues that do appear together were checked with cvd_check.py:

    accent / ceiling      normal 39.2   deutan 38.4   protan 19.3   PASS

Ball colours are never drawn from this palette — a swatch for the blue ball is
drawn in that ball's actual RGB, because there it is data, not encoding.
"""

from flat2d_style import (  # noqa: F401
    SURFACE, INK, INK2, MUTED, GRID, AXIS, RULE, NEUTRAL, use_style, despine)

ACCENT = '#8fbadb'    # the measured quantity in each panel
CEILING = '#17344e'   # the per-colour fitted map, the best achievable
FLOOR = '#0b0b0b'     # the red-ball reference line

# ordered weakest to strongest by how much information each method uses
METHODS = ['nomap', 'linear', 'interp', 'shared', 'fitted']
METHOD_LABEL = {
    'nomap': 'no map',
    'linear': 'predicted: linear in colour',
    'interp': 'predicted: blend of neighbours',
    'shared': 'predicted: one shared map',
    'fitted': 'fitted on this colour (ceiling)',
}
