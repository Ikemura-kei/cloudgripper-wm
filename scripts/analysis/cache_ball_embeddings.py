import os, sys, numpy as np, torch
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, 'scripts/analysis')
from probe_ball_latent import build_dataset, extract, SETS
import stable_worldmodel as swm

H, dev, N = 6, 'cuda', 4000
m = swm.wm.utils.load_pretrained(os.path.abspath(
    'datasets/checkpoints/lewm_ball-26-10-08-23-51-41-rzr/weights_epoch_100.pt')).to(dev).eval()
m.requires_grad_(False)
out, norms = {}, None
for label, name in SETS.items():
    ds, norms = build_dataset(name, os.getcwd(), H, 336, norms)
    cls, emb, pred, pos, spd = extract(m, ds, N, H, 32, dev)
    for k, v in zip(('cls', 'proj', 'pred', 'pos', 'spd'), (cls, emb, pred, pos, spd)):
        out[f'{label}|{k}'] = v
    print(f'{label:9s} {len(spd):5d} windows', flush=True)
np.savez_compressed('outputs/ball_probe/emb_cache.npz', **out)
print('cached ->', os.path.getsize('outputs/ball_probe/emb_cache.npz') / 1e6, 'MB')
