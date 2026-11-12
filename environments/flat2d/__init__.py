from gymnasium.envs.registration import register

register(
    id="flat2d/Ball-v0",
    entry_point="environments.flat2d.flat2d_ball:Flat2DBall",
)
