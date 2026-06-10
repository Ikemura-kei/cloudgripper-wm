from abc import ABC, abstractmethod

import numpy as np

DEFAULT_HOME_POS = np.array([0.1, 0.1, 1.0, 0.0, 0.0], dtype=np.float32)


class Task(ABC):
    @abstractmethod
    def compute_reward(self, obs: dict, action: np.ndarray, info: dict) -> float: ...

    @abstractmethod
    def check_success(self, obs: dict, info: dict) -> bool: ...

    def check_terminated(self, obs: dict, info: dict) -> bool:
        return False

    def get_task_info(self, obs: dict) -> dict:
        return {}

    def home_pos(self) -> np.ndarray:
        return DEFAULT_HOME_POS.copy()
