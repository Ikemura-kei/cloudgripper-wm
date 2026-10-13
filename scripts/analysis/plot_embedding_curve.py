"""Show why position is 'curved' in the embedding rather than linear.

Panel choices are deliberate, not arbitrary:
  A  the three LEAST oscillatory of the 192 dimensions -- the best case for a
     straight-line relationship. If even these reverse direction repeatedly,
     the point does not depend on which dimension is shown.
  B  Euclidean distance between pairs of embeddings against how far apart the
     two balls are. The reference slope is fitted on pairs closer than 10 mm,
     so it reflects the local behaviour rather than one histogram bin.
  C  a PCA projection, which is only a 2-D shadow of a path that needs ~27
     dimensions to describe -- shown for shape, not as a faithful depiction.
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from sklearn.decomposition import PCA

AX, SMOOTH = 1, 4000
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#dcdcd8', '#fcfcfb'

def style(ax, t, xl, yl):
    ax.set_title(t, fontsize=10.5, color=INK, pad=10, loc='left')
    ax.set_xlabel(xl, fontsize=9, color=INK2); ax.set_ylabel(yl, fontsize=9, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(True, color=GRID, lw=.7); ax.set_axisbelow(True); ax.set_facecolor(SURF)
    for s in ax.spines.values(): s.set_color(GRID)

d = np.load('outputs/ball_probe/emb_cache.npz')
E = d['train|proj'].reshape(-1, 192); P = d['train|pos'][:, :, AX].reshape(-1) * 1000
o = np.argsort(P); E, P = E[o], P[o]

sm = np.apply_along_axis(lambda v: np.convolve(v, np.ones(SMOOTH)/SMOOTH, 'valid'), 0, E)
turns = (np.diff(np.sign(np.diff(sm, axis=0)), axis=0) != 0).sum(0)
calm = np.argsort(turns)[:3]
Psm = P[SMOOTH//2: SMOOTH//2 + len(sm)]

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8), facecolor=SURF)

for i, c in zip(calm, (C1, C2, C3)):
    ax[0].plot(Psm, sm[:, i], lw=1.6, color=c,
               label=f'dim {i} — turns {turns[i]}x')
style(ax[0], f'A  The 3 steadiest of 192 numbers (median dim turns {int(np.median(turns))}x)',
      'ball position (mm)', 'value of that number')
ax[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2)

rng = np.random.default_rng(0)
i, j = rng.integers(0, len(P), 60000), rng.integers(0, len(P), 60000)
dp, de = np.abs(P[i] - P[j]), np.linalg.norm(E[i] - E[j], axis=1)
b = np.linspace(0, 240, 60); k = np.digitize(dp, b)
m = np.array([de[k == t].mean() if (k == t).sum() > 20 else np.nan for t in range(1, 60)])
near = dp < 10
slope = np.sum(de[near] * dp[near]) / np.sum(dp[near] ** 2)   # fitted on close pairs
ax[1].scatter(dp[::25], de[::25], s=3, alpha=.12, color=C1, lw=0)
ax[1].plot(b[1:], m, color=C2, lw=2.4, label='average')
ax[1].plot([0, 240], [0, slope * 240], color=INK2, ls='--', lw=1.3,
           label='if distance grew proportionally')
style(ax[1], 'B  Distance in embedding vs distance on the track',
      'how far apart the two balls are (mm)', 'how far apart the embeddings are')
ax[1].set_ylim(0, 45); ax[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2)

p = PCA(3).fit(E); p3 = p.transform(E)
sc = ax[2].scatter(p3[:, 0], p3[:, 1], c=P, cmap='viridis', s=4, lw=0)
style(ax[2], f'C  A 2-D shadow of the path '
             f'({p.explained_variance_ratio_[:2].sum()*100:.0f}% of the variation)',
      'component 1', 'component 2')
plt.colorbar(sc, ax=ax[2], label='ball position (mm)')

fig.suptitle('Why a straight-line probe struggles: position curves through the embedding',
             fontsize=12.5, color=INK, y=.99)
fig.tight_layout(rect=[0, 0, 1, .95])
fig.savefig('outputs/ball_probe/embedding_curve.png', dpi=150, facecolor=SURF)
print(f'monotonic dimensions: {(turns == 0).sum()} of 192')
print(f'steadiest dims {calm} turn {turns[calm]} times; median {int(np.median(turns))}')
print(f'local slope {slope:.4f} units/mm -> proportional growth would predict '
      f'{slope*200:.1f} at 200 mm; actual {de[dp>200].mean():.1f}')
