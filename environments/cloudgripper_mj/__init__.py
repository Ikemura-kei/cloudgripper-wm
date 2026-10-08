from gymnasium.envs.registration import register

register(
    id="cloudgripper_mj/Tracking-v0",
    entry_point="environments.cloudgripper_mj.cloudgripper_mj_tracking:CloudgripperMuJoCoTracking",
)

register(
    id="cloudgripper_mj/Cube-v0",
    entry_point="environments.cloudgripper_mj.cloudgripper_mj_cube:CloudgripperMuJoCoCube",
)

register(
    id="cloudgripper_mj/Ball-v0",
    entry_point="environments.cloudgripper_mj.cloudgripper_mj_ball:CloudgripperMuJoCoBall",
)
