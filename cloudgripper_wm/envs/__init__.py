from gymnasium.envs.registration import register

register(
    id="cloudgripper/Gripper-v0",
    entry_point="cloudgripper_wm.envs.cloudgripper_env:CloudGripperEnv",
    max_episode_steps=100,
)

_TASKS = {
    "cloudgripper/CubePush-v0": "cube_push",
    "cloudgripper/CubeStack-v0": "cube_stack",
    "cloudgripper/RopeManip-v0": "rope_manip",
}

for _env_id, _task_name in _TASKS.items():
    register(
        id=_env_id,
        entry_point="cloudgripper_wm.envs.cloudgripper_env:CloudGripperEnv",
        max_episode_steps=100,
        kwargs={"task": _task_name},
    )
