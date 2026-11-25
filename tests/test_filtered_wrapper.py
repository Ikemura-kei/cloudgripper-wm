"""FilteredCloudGripperWrapper on the mock robot with a made-up scene: a free action goes through, a blocked one is zeroed."""
from types import SimpleNamespace

import numpy as np

from cloudgripper_wm import collision_filter as cf
from cloudgripper_wm.envs.cloudgripper_env import CloudGripperEnv
from cloudgripper_wm.envs.filtered_wrapper import FilteredCloudGripperWrapper
from cloudgripper_wm.envs.robot_pool import RobotPool


def test_blocked_action_is_zeroed():
    keep = {k: getattr(cf, k) for k in list(cf.CELL20) + ["_FRAME"]}
    half = 5 * cf.MM
    cube = SimpleNamespace(status="on_glass", on=None, height_mm=10, inferred_xy=None,
                           footprint_xy=np.array([[0.5, 0.5]]) + half * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]))
    scene = dict(cube_blue=cube)
    tracker = SimpleNamespace(start=lambda b, t: scene, step=lambda b, t: scene, refresh=lambda b, t: scene)
    RobotPool.configure(["robot20"])
    env = FilteredCloudGripperWrapper(CloudGripperEnv(use_mock=True, dwell_time=0, reset_dwell_time=0, max_delta=0.5), tracker=tracker)
    try:
        env.reset()
        u = env.unwrapped
        assert u._target_pos[4] == np.float32(0.1)  # the home pose's gripper 0.0 is raised to cell 20's closed value
        env.step(np.float32([0.2, 0.2, 0, 0, 0]))  # sideways in the air: free
        assert np.allclose(u._target_pos[:2], [0.7, 0.7]) and env.n_blocked == 0
        env.step(np.float32([-0.2, -0.2, 0, 0, 0]))  # back over the cube (the home pose is the glass centre)
        env.step(np.float32([0, 0, -0.4, 0, 0]))  # down to 16 mm, above the 10 mm cube: free
        env.step(np.float32([0, 0, -0.6, 0, 0]))  # down onto the cube: blocked, pose held
        assert env.n_blocked == 1 and np.isclose(u._target_pos[2], 0.6)
    finally:
        env.close()
        for k, val in keep.items():
            setattr(cf, k, val)
