"""Random policy that never moves the gripper.

Identical to RandomPolicy except the last action dimension (gripper delta)
is always zeroed out, so the gripper aperture stays fixed at whatever it was
set to on reset.
"""

import numpy as np
from stable_worldmodel.policy import BasePolicy


class NoGripRandomPolicy(BasePolicy):
    """RandomPolicy with the gripper action dimension always zeroed.

    Args:
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(self, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(seed)

    def get_action(self, obs, **kwargs) -> np.ndarray:
        action = self.env.action_space.sample()
        action[..., -1] = 0.0
        return action
