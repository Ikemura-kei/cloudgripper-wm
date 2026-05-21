import numpy as np

from cloudgripper_wm.tasks.base import Task


class RopeManipTask(Task):
    def compute_reward(self, obs: dict, action: np.ndarray, info: dict) -> float:
        return 0.0

    def check_success(self, obs: dict, info: dict) -> bool:
        return False
