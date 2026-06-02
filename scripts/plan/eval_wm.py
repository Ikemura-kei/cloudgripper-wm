"""Script to evaluate a World Model using MPC on a dataset of episodes."""

import os

os.environ['MUJOCO_GL'] = 'egl'

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from cloudgripper_wm.world import CloudGripperWorld

def img_transform(cfg, dtype=torch.float32):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = 'episode_idx'

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.get('cache_dir', None),
    )
    return dataset


def _augment_video_with_goal(
    video_path: Path,
    goal_frame: np.ndarray,
    null_frames: list[np.ndarray] | None = None,
) -> None:
    """Re-encode a video with the goal image as a fixed left panel."""
    import cv2
    import imageio

    if not video_path.exists():
        return

    reader = imageio.get_reader(str(video_path))
    fps = reader.get_meta_data().get('fps', 15)
    frames = [np.asarray(f) for f in reader]
    reader.close()

    if not frames:
        return

    # Resize goal to match video frame height
    vid_h = frames[0].shape[0]
    goal_h, goal_w = goal_frame.shape[:2]
    panel_w = int(goal_w * vid_h / goal_h)
    panel_w += panel_w % 2  # libx264 requires even width
    goal_panel = cv2.resize(goal_frame, (panel_w, vid_h))
    cv2.putText(goal_panel, 'GOAL', (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 0), 2, cv2.LINE_AA)

    out_fps = fps / 2
    freeze_frames = int(out_fps * 5)  # 5 seconds of the last frame

    out_path = video_path.parent / (video_path.stem + '_with_goal.mp4')
    writer = imageio.get_writer(str(out_path), fps=out_fps, codec='libx264',
                                macro_block_size=1)
    for frame in (null_frames or []):
        if frame.shape[0] != vid_h:
            frame = cv2.resize(frame, (frame.shape[1], vid_h))
        writer.append_data(np.concatenate([goal_panel, frame], axis=1))
    for frame in frames:
        writer.append_data(np.concatenate([goal_panel, frame], axis=1))
    last = np.concatenate([goal_panel, frames[-1]], axis=1)
    for _ in range(freeze_frames):
        writer.append_data(last)
    writer.close()
    print(f'[video] goal-augmented → {out_path}')


def _show_goal_images(dataset, eval_episodes, eval_start_idx, goal_offset):
    """Save goal frames to /tmp and show via matplotlib (non-blocking).

    Uses matplotlib instead of cv2.imshow to avoid conflicting with the
    env's cv2 display thread (cv2.waitKey pumps a shared Qt event loop).
    """
    import cv2
    import matplotlib
    import matplotlib.pyplot as plt

    ep_col   = dataset.get_col_data('episode_idx').astype(int)
    step_col = dataset.get_col_data('step_idx').astype(int)

    goal_imgs = []
    for ep, start in zip(eval_episodes, eval_start_idx):
        goal_step = int(start) + goal_offset
        row_mask = (ep_col == int(ep)) & (step_col == goal_step)
        row_idx = int(np.flatnonzero(row_mask)[0])
        jpeg = dataset.get_row_data([row_idx])['pixels'][0]
        arr = np.frombuffer(jpeg, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        goal_imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    label = (f'Goal  ep={list(eval_episodes)}  '
             f'start={list(eval_start_idx.astype(int))}  +{goal_offset} steps')
    print(f'[goal] {label}')

    fig, axes = plt.subplots(1, len(goal_imgs), squeeze=False,
                             figsize=(4 * len(goal_imgs), 4))
    for ax, img in zip(axes[0], goal_imgs):
        ax.imshow(img)
        ax.set_title(label, fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    save_path = '/tmp/lewm_goal.png'
    plt.savefig(save_path, dpi=100)
    print(f'[goal] saved → {save_path}')
    plt.ion()
    plt.show()
    plt.pause(0.5)   # give the window time to render before eval starts


@hydra.main(version_base=None, config_path='./config', config_name='cloudgripper')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    # create world environment — CloudGripperWorld handles RobotPool.configure()
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    from cloudgripper_wm.world import CloudGripperWorld
    from cloudgripper_wm.envs.robot_pool import RobotPool
    import cloudgripper_wm.envs  # triggers gymnasium.register()

    num_envs = cfg.world.num_envs
    robot_names = list(cfg.world.get('robot_names', [f'robot{i+1}' for i in range(num_envs)]))
    RobotPool.configure(robot_names)
    _skip = {'robot_names', 'image_shape'}
    image_shape = tuple(cfg.world.get('image_shape', [224, 224]))
    world = swm.World(**{k: v for k, v in cfg.world.items() if k not in _skip},
                      image_shape=image_shape)

    # create the transform
    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {
        'pixels': img_transform(cfg, img_dtype),
        'goal': img_transform(cfg, img_dtype),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = get_dataset(cfg, cfg.eval.dataset_name)  # separate instance — get_col_data caches columns as a side effect, which breaks load_chunk
    col_name = 'episode_idx'
    ep_indices, _ = np.unique(
        stats_dataset.get_col_data(col_name), return_index=True
    )

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != 'action':
            process[f'goal_{col}'] = process[col]

    # -- run evaluation
    policy = cfg.get('policy', 'random')

    if policy != 'random':
        model = swm.wm.utils.load_pretrained(cfg.policy)
        if cfg.get('bf16', False):
            model = model.to(torch.bfloat16)
        model = model.to('cuda')
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        if cfg.get('compile', False):
            encoder_attr = (
                'backbone' if hasattr(model, 'backbone') else 'encoder'
            )
            setattr(
                model,
                encoder_attr,
                torch.compile(getattr(model, encoder_attr)),
            )
            model.predictor = torch.compile(model.predictor)
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)

        # -- logging wrapper: print cost + actions after every MPC replan --
        _mpc_step = [0]
        _orig_solve = solver.solve

        def _solve_with_log(info_dict, init_action=None):
            result = _orig_solve(info_dict, init_action)
            _mpc_step[0] += 1
            cost = result['costs'][0] if result['costs'] else float('nan')
            # result['actions']: (n_envs, horizon, action_dim) — normalised
            actions_norm = result['actions'][0].numpy()          # (horizon, 5)
            future = process['action'].inverse_transform(actions_norm) if 'action' in process else actions_norm
            print(f'\n── MPC replan {_mpc_step[0]:02d}  best_cost={cost:.5f} ──')
            print(f"{'t':>3}  {'Δx':>7}  {'Δy':>7}  {'Δz':>7}  {'Δrot':>7}  {'Δgrip':>7}")
            for i, a in enumerate(future):
                print(f'{i:3d}  {a[0]:7.4f}  {a[1]:7.4f}  {a[2]:7.4f}  {a[3]:7.4f}  {a[4]:7.4f}')
            return result

        solver.solve = _solve_with_log
        # ------------------------------------------------------------------

        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy
        ).parent
        if cfg.policy != 'random'
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    # Map each dataset row’s episode_idx to its max_start_idx
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for evaluation.')

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    ep_col_all   = dataset.get_col_data('episode_idx')
    step_col_all = dataset.get_col_data('step_idx')
    eval_episodes  = ep_col_all[random_episode_indices]
    eval_start_idx = step_col_all[random_episode_indices]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for evaluation.'
        )

    # -- show goal images while planning executes -------------------------
    _show_goal_images(dataset, eval_episodes, eval_start_idx, cfg.eval.goal_offset_steps)

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)
    print(
        f'[eval] saving videos to {results_path.resolve()} '
        '(one env_{i}.mp4 per env)'
    )

    autocast_ctx = torch.autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=cfg.get('bf16', False),
    )

    if cfg.get('compile', False):
        print('Warming up compiled model...')
        warmup_autocast_ctx = torch.autocast(
            device_type='cuda',
            dtype=torch.bfloat16,
            enabled=cfg.get('bf16', False),
        )
        with warmup_autocast_ctx:
            n = world.num_envs
            world.evaluate(
                dataset=dataset,
                start_steps=eval_start_idx.tolist()[:n],
                goal_offset=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=eval_episodes.tolist()[:n],
                callables=OmegaConf.to_container(
                    cfg.eval.get('callables'), resolve=True
                ) if cfg.eval.get('callables') is not None else None,
                video=results_path,
            )
        print('Warmup done.')

    # -- reset + null actions before planning so the robot settles --------
    N_NULL = 3
    print(f'[reset] resetting env and capturing {N_NULL} null frames...')
    world.reset()
    # Capture the reset-position frame N_NULL times; world.evaluate() will
    # issue its own reset immediately after, so the robot stays at home.
    reset_frame = world.infos['pixels'][:, 0].copy()  # (num_envs, H, W, C)
    null_frames: list[np.ndarray] = [reset_frame] * N_NULL
    print('[reset] done.')
    # ----------------------------------------------------------------------

    start_time = time.time()
    with autocast_ctx:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(
                cfg.eval.get('callables'), resolve=True
            ) if cfg.eval.get('callables') is not None else None,
            video=results_path,
        )
    end_time = time.time()

    print(metrics)
    print(f'[eval] videos saved to {results_path.resolve()}')

    # -- augment each env video: prepend the goal frame as a left panel ----
    for env_idx in range(world.num_envs):
        _augment_video_with_goal(
            results_path / f'env_{env_idx}.mp4',
            goal_frame=world.infos['goal'][env_idx, 0],  # (H, W, C) uint8
            null_frames=[nf[env_idx] for nf in null_frames],
        )

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open('a') as f:
        f.write('\n')  # separate from previous runs

        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n')

        f.write('==== RESULTS ====\n')
        f.write(f'metrics: {metrics}\n')
        f.write(f'evaluation_time: {end_time - start_time} seconds\n')


if __name__ == '__main__':
    run()