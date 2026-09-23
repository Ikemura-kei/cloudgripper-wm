"""Watch the CloudGripper MuJoCo env run under random actions in the live viewer.

The env itself only renders offscreen (render_modes: ["rgb_array"]), so this
attaches a passive viewer to the underlying MjModel/MjData to get a window.

Run from the repo root:
    uv run python environments/mj_cloudgripper/viz_env.py
"""

import time

import gymnasium as gym
import mujoco
import mujoco.viewer

import environments.mj_cloudgripper  # noqa: F401  (triggers gymnasium registration)


def main() -> None:
    env = gym.make("cloudgripper_mujoco/Tracking-v0", height=224, width=224)
    env.reset(seed=0)

    inner = env.unwrapped
    viewer = mujoco.viewer.launch_passive(inner.model, inner.data)
    try:
        while viewer.is_running():
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            viewer.sync()
            time.sleep(inner._control_timestep)
            if terminated or truncated:
                env.reset()
    finally:
        viewer.close()
        env.close()


if __name__ == "__main__":
    main()
