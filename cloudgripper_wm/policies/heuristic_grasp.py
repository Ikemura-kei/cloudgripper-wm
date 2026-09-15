
import numpy as np
from stable_worldmodel.policy import BasePolicy


class HeuristicGraspPolicy(BasePolicy):
    """Follows the env's own expert trajectory.
    
     Env's using this policy must implement a normalized expert target
     waypoint and the current commanded pose in the observation.
    """

    def get_action(self, obs, **kwargs) -> np.ndarray:
        """Returns the delta action guiding towards the target pose of
        the current phase. Actions are kept within the upper and lower
        limits of the action space.
        
        args:
            obs: current observation
        returns:
            ndarray of delta action
        """
        expert_target = np.asarray(obs["expert_target"])[:, 0, :]
        state = np.asarray(obs["state"])[:, 0, :]
        delta = expert_target - state
        return np.clip(
            delta, self.env.action_space.low, self.env.action_space.high
        ).astype(np.float32)
