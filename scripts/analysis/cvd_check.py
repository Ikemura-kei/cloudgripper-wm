"""Colour-vision-deficiency separation check for categorical palettes.

Implements the two gates the dataviz method requires, so palette choices are
computed rather than eyeballed:

    normal-vision OKLab dE >= 15   (full-colour readers can tell the pair apart)
    simulated-CVD OKLab dE >= 8    (deuteranope / protanope readers can too)

OKLab per Ottosson (2020); dichromat simulation per Vienot, Brettel & Mollon
(1999), the standard LMS-projection construction. dE is Euclidean in OKLab
scaled x100, matching the convention the method's own validator uses.
"""

import numpy as np

_LIN = lambda c: np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
_SRGB = lambda c: np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)

# Vienot et al. (1999) linear-sRGB -> LMS
_RGB2LMS = np.array([[17.8824, 43.5161, 4.11935],
                     [3.45565, 27.1554, 3.86714],
                     [0.0299566, 0.184309, 1.46709]])
_LMS2RGB = np.linalg.inv(_RGB2LMS)
_PROTAN = np.array([[0.0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]])
_DEUTAN = np.array([[1, 0, 0], [0.494207, 0.0, 1.24827], [0, 0, 1]])

# Ottosson (2020) linear-sRGB -> OKLab
_RGB2L = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
_L2LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])


def hex_to_linear(h: str) -> np.ndarray:
    h = h.lstrip('#')
    return _LIN(np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]))


def simulate(lin: np.ndarray, kind: str) -> np.ndarray:
    """Project a linear-sRGB colour onto a dichromat's gamut."""
    mat = _PROTAN if kind == 'protan' else _DEUTAN
    lms = _RGB2LMS @ lin
    return np.clip(_LMS2RGB @ (mat @ lms), 0, 1)


def oklab(lin: np.ndarray) -> np.ndarray:
    return _L2LAB @ np.cbrt(_RGB2L @ lin)


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(oklab(a) - oklab(b)) * 100)


def report(pairs: dict) -> None:
    print(f'{"pair":<26}{"normal":>9}{"deutan":>9}{"protan":>9}   verdict')
    for label, (h1, h2) in pairs.items():
        a, b = hex_to_linear(h1), hex_to_linear(h2)
        n = delta_e(a, b)
        d = delta_e(simulate(a, 'deutan'), simulate(b, 'deutan'))
        p = delta_e(simulate(a, 'protan'), simulate(b, 'protan'))
        ok = n >= 15 and min(d, p) >= 8
        why = [] if ok else ([f'normal {n:.1f}<15'] if n < 15 else []) + \
                            ([f'CVD {min(d, p):.1f}<8'] if min(d, p) < 8 else [])
        print(f'{label:<26}{n:>9.1f}{d:>9.1f}{p:>9.1f}   {"PASS" if ok else "FAIL: " + ", ".join(why)}')


if __name__ == '__main__':
    report({
        'semantic red/green': ('#e34948', '#008300'),
        'palette blue/orange': ('#2a78d6', '#eb6834'),
        'palette blue/red': ('#2a78d6', '#e34948'),
    })
