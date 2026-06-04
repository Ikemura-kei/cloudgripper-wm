"""Random policy with a zero-action warmup at the start of each episode.

For the first `n_warmup` steps after reset() is called, the policy outputs
zeros. After that it behaves identically to RandomPolicy.
"""

import numpy as np
from stable_worldmodel.policy import BasePolicy


class WarmupRandomPolicy(BasePolicy):
    """RandomPolicy that outputs zero actions for the first n_warmup steps.

    Call reset() between episodes (collect_cloudgripper.py does this).

    Args:
        n_warmup: Number of zero-action steps at the start of each episode.
        seed: Optional RNG seed.
    """

    def __init__(
        self,
        n_warmup: int = 2,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_warmup = n_warmup
        self.rng = np.random.default_rng(seed)
        self._steps_since_reset: int = 0

    def get_action(self, obs, **kwargs) -> np.ndarray:
        if self._steps_since_reset < self.n_warmup:
            action = np.zeros_like(self.env.action_space.sample())
        else:
            action = self.env.action_space.sample()
        self._steps_since_reset += 1
        return action

    def reset(self) -> None:
        self._steps_since_reset = 0
