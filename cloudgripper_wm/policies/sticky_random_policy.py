"""Sticky random policy — holds a sampled action for several steps with small noise.

Each "macro-action" is sampled once from the action space, then repeated for
2–5 steps (uniform random). At every step a small Gaussian noise is added to
the base action before clipping to the action space bounds.
"""

import numpy as np
from stable_worldmodel.policy import BasePolicy


class StickyRandomPolicy(BasePolicy):
    """Random policy with action persistence and per-step noise.

    A base action is sampled from the environment's action space and held for
    `repeat` steps, where `repeat` ~ Uniform{min_repeat, …, max_repeat}.
    At each step, independent Gaussian noise (std=noise_std) is added and the
    result is clipped to the action space bounds.

    Args:
        min_repeat: Minimum number of steps to repeat a sampled action.
        max_repeat: Maximum number of steps to repeat a sampled action.
        noise_std: Standard deviation of per-step additive Gaussian noise.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        min_repeat: int = 2,
        max_repeat: int = 5,
        noise_std: float = 0.1,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_repeat = min_repeat
        self.max_repeat = max_repeat
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self._base_action: np.ndarray | None = None
        self._steps_remaining: int = 0

    def get_action(self, obs, **kwargs) -> np.ndarray:
        if self._steps_remaining <= 0:
            self._base_action = self.env.action_space.sample()
            self._steps_remaining = int(
                self.rng.integers(self.min_repeat, self.max_repeat + 1)
            )

        noise = self.rng.normal(0.0, self.noise_std,
                                size=self._base_action.shape).astype(np.float32)
        action = np.clip(
            self._base_action + noise,
            self.env.action_space.low,
            self.env.action_space.high,
        )

        self._steps_remaining -= 1
        return action

    def reset(self) -> None:
        """Call between episodes to force a new action sample."""
        self._base_action = None
        self._steps_remaining = 0
