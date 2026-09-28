"""
"""

import time

import gymnasium as gym
import mujoco
import mujoco.viewer

import environments.cloudgripper_mj  # noqa: F401  (triggers gymnasium registration)


def main() -> None:
    env = gym.make("cloudgripper_mj/Cube-v0", height=224, width=224)
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
