
import numpy as np
from stable_worldmodel.policy import BasePolicy


class HeuristicGraspPolicy(BasePolicy):
    """Follows the env's own expert trajectory.

     Env's using this policy must implement a normalized expert target
     waypoint in the observation, and its action space must accept that
     waypoint directly (absolute positioning, not delta/step-wise).
    """

    def get_action(self, obs, **kwargs) -> np.ndarray:
        """Returns the expert's absolute target pose for the current
        phase, clipped to the action space's bounds.

        args:
            obs: current observation
        returns:
            ndarray of absolute target pose
        """
        expert_target = np.asarray(obs["expert_target"])[:, 0, :]
        return np.clip(
            expert_target, self.env.action_space.low, self.env.action_space.high
        ).astype(np.float32)
