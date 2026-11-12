"""Shared settings for the colour-generalisation study.

The question: the red encoder, shown a recoloured ball, produces embeddings
that an affine map sends back to the red-ball embeddings (R2 0.986, measured
on the red/green pair). Is that map a *predictable function of the colour
change*? If it is, a new colour needs no fitting and no new world model.

Everything here runs against the single red model from the rolling study
(`rolling_common.RUNS['red']`). No training happens in this study at all.

Collection design
-----------------
All colours are collected at SEED, which differs from the seed the training
datasets used. Two consequences, both wanted:

  * none of these episodes were seen by the red model during training, so the
    reference embeddings are held-out ones;
  * ball colour is a fixed constructor kwarg and consumes no variation draws,
    so every colour replays the identical episodes frame for frame. Window *i*
    is the same scene in all ten datasets, differing only in the ball's pixels.

The second property is what makes a per-colour map well posed: source and
target rows describe the same moment.

Colour set
----------
Ten saturated hues spanning the RGB cube, avoiding anything near the wall
(white) or floor (black) where the ball would stop being visible. The three
primaries plus yellow pin down a linear model of the map; the rest are the
held-out probes that test it. `orange` and `rose` sit between red and a
measured colour, `azure` and `spring` sit far from red — so the probes cover
both interpolation and extrapolation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / '.stable_worldmodel' / 'datasets'

# The study exists in two parallel versions, differing only in how images are
# normalised before the encoder.
#
#   imagenet   the original. Inherited from the upstream config, where it makes
#              sense because the backbone is pretrained on ImageNet. Here the
#              encoder is trained from scratch, and the per-channel means
#              (0.485, 0.456, 0.406) put green ~0.22 below blue in every pixel
#              of a scene whose raw frames are byte-identical under a green/blue
#              swap. Measured consequence: blue balls are squashed ~2.5x harder
#              than green ones, and the effect survives a change of seed.
#   symmetric  0.5 / 0.25 on every channel, so the network's input really is
#              swap invariant. The green/blue asymmetry collapses from 3.07x to
#              1.11x and its direction becomes a coin toss across seeds, while
#              the red-vs-everything asymmetry is untouched.
#
# Both are kept: the first is what the published results were computed with,
# the second is what the scene actually implies. Separate caches, separate
# model runs, nothing shared.
STATS = 'imagenet'
CACHE = REPO / 'outputs' / 'colour_cache'
OUT = REPO / 'outputs' / 'colour_study'
_CACHE = {'imagenet': REPO / 'outputs' / 'colour_cache',
          'symmetric': REPO / 'outputs' / 'colour_cache_sym'}
_OUT = {'imagenet': REPO / 'outputs' / 'colour_study',
        'symmetric': REPO / 'outputs' / 'colour_study_sym'}
# which trained model plays the part of "the red encoder" in each version
REF_RUN = {'imagenet': 'red', 'symmetric': 'red_sym_s1'}


def use_stats(name: str) -> str:
    """Point the module at one of the two versions. Call once, early."""
    global STATS, CACHE, OUT
    if name not in _CACHE:
        raise ValueError(f'stats={name!r}; expected one of {sorted(_CACHE)}')
    STATS, CACHE, OUT = name, _CACHE[name], _OUT[name]
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    return name

SEED = 1234
EPISODES = 200

# name -> RGB 0-255
COLOURS: dict[str, tuple[int, int, int]] = {
    'red':     (255, 0, 0),      # the colour the model was trained on
    'green':   (0, 255, 0),
    'blue':    (0, 0, 255),
    'yellow':  (255, 255, 0),
    'cyan':    (0, 255, 255),
    'magenta': (255, 0, 255),
    'orange':  (255, 128, 0),
    'spring':  (0, 255, 128),
    'azure':   (0, 128, 255),
    'rose':    (255, 0, 128),
    # --- extension to 30 colours -------------------------------------------
    # The ten above are all fully saturated, which leaves saturation and
    # lightness untested as factors. These twenty were chosen by farthest-point
    # sampling over a 32-step RGB grid, seeded with the ten above, after
    # discarding anything within 110 of white or black in RGB — a ball that
    # close to the wall or the floor stops being visible against it. The set
    # spans saturation 0.00 (a grey ball) to 1.00 and drops the minimum
    # pairwise distance from 127 to 78.
    'c11':    (64, 64, 64),
    'c12':    (160, 192, 192),
    'c13':    (128, 32, 192),
    'c14':    (128, 192, 32),
    'c15':    (192, 96, 96),
    'c16':    (0, 64, 160),
    'c17':    (0, 160, 64),
    'c18':    (128, 0, 0),
    'c19':    (64, 160, 160),
    'c20':    (224, 96, 224),
    'c21':    (224, 224, 96),
    'c22':    (160, 0, 96),
    'c23':    (160, 96, 0),
    'c24':    (64, 0, 128),
    'c25':    (64, 128, 0),
    'c26':    (0, 192, 192),
    'c27':    (64, 64, 224),
    'c28':    (64, 224, 64),
    'c29':    (96, 96, 128),
    'c30':    (96, 224, 224),
}

REFERENCE = 'red'

# The colours whose maps are fitted and used to build the linear model of
# b_c. Red is the reference (its map is the identity by construction), so the
# model is pinned by green, blue and yellow — three independent colour
# changes, which is exactly enough to determine a 3-column matrix.
# Kept from the original ten-colour design, where the hypothesis was tested by
# fitting on exactly three colours and predicting six. With thirty colours the
# analysis uses leave-one-colour-out instead, which uses all of them; these two
# lists survive only for the scripts that still reference them.
BASIS = ['green', 'blue', 'yellow']
PROBES = ['cyan', 'magenta', 'orange', 'spring', 'azure', 'rose']


def dataset_name(colour: str) -> str:
    return f'colour_ball_{colour}'


def rgb(colour: str) -> np.ndarray:
    return np.asarray(COLOURS[colour], dtype=np.float64)


def delta(colour: str) -> np.ndarray:
    """Colour change relative to the training colour, scaled to [-1, 1]."""
    return (rgb(colour) - rgb(REFERENCE)) / 255.0


# --- dataset plumbing -------------------------------------------------------
# Reuses the rolling study's constants (history 3, frameskip 5, 224 px) because
# the model under test is that study's red model and it must be fed exactly
# what it was trained on.

def build_dataset(colour: str, norms: dict | None = None, stats: str | None = None):
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from stable_pretraining import data as dt

    import rolling_common as R

    ds = swm.data.load_dataset(
        str(DATA / f'{dataset_name(colour)}.lance'),
        transform=None, num_steps=R.HISTORY + 1, frameskip=R.FRAMESKIP,
        keys_to_load=['pixels', 'action', 'state', 'ball_pos', 'ball_speed'])
    R.IMG_STATS['imagenet'] = dt.dataset_stats.ImageNet
    stats = R.IMG_STATS[stats or STATS]
    tfs = [dt.transforms.Compose(
        dt.transforms.ToImage(**stats, source='pixels', target='pixels'),
        dt.transforms.Resize(R.IMG_SIZE, source='pixels', target='pixels'))]
    if norms is None:
        raise ValueError('pass norms from training_norms() — the model was '
                         'trained with the training dataset\'s statistics')
    tfs.extend(norms.values())
    ds.transform = spt.data.transforms.Compose(*tfs)
    return ds


def training_norms() -> dict:
    """Normalisers for the action/state columns, from the training dataset.

    These are column statistics, not image statistics, so they do not depend on
    STATS — but the dataset must still be built with the matching image stats
    for the rest of the pipeline to line up.
    """
    import rolling_common as R
    _, norms = R.build_dataset('red', stats=STATS)
    return norms


def map_split(ds, test_frac: float = 0.4, seed: int = 7):
    """(fit, test) window indices, split by episode so no episode straddles.

    The map is fitted on one set of episodes and scored on another. Splitting
    by window instead would put near-duplicate frames on both sides and make
    every map look better than it is.
    """
    clip_eps = np.asarray([ep for ep, _ in ds.clip_indices])
    episodes = np.unique(clip_eps)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(episodes))
    n_test = max(1, int(round(test_frac * len(episodes))))
    test_eps = set(episodes[perm[:n_test]].tolist())
    is_test = np.array([e in test_eps for e in clip_eps])
    return np.where(~is_test)[0], np.where(is_test)[0]
