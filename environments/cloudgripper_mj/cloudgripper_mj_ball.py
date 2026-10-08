"""CloudGripper MuJoCo ball task — a ball crossing the scene at constant speed.

Built for latent-space studies of world models (LeWM / JEPA family): the only
thing that exists in the scene besides the empty cell is a ball travelling
left->right across the camera image, at a speed drawn per episode. Everything
else is held fixed, so a dataset of these episodes varies in essentially one
dimension.

The scene is `cloudgripper_static_scene.xml` — the cell without the robot's
moving parts, so there is no arm in frame and nothing in the episode responds
to actions. Actions are still sampled and recorded (the training pipeline
expects an action column) but drive nothing, which is exactly the setup for
asking whether a world model separates controllable from uncontrollable
dynamics.

The ball is a *mocap* body rather than a dynamic one. MuJoCo's integrator
leaves mocap bodies alone, so the ball's velocity is exactly the sampled one
for the whole episode — friction, rolling resistance, contact and gravity
cannot perturb it, which is the point ("constant speed", with friction not
allowed to matter). It sits at a height that makes it look like it rests on
the ground plate, but never touches anything: its geom has
``contype=0``/``conaffinity=0``. Flat-shaded sphere, so no visible spin — it
reads as a ball sliding along the floor, which the task allows.

Ground-truth factors land in the step info (and hence as dataset columns) as
``ball_pos``/``ball_speed``/``ball_color``, for probing a learned latent
against what actually generated the frames. They are deliberately kept out of
the ``state`` observation so a model conditioned on state can't read the
answer off its input.
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from dm_control import mjcf
from gymnasium.spaces import Box
from ogbench.manipspace.envs.env import CustomMuJoCoEnv
from stable_worldmodel import spaces as swm_spaces


# Only the speed is resampled per episode. The colour stays at its init_value
# (red) unless a caller asks for 'ball.color' explicitly, e.g. via the
# collection script's `variation=[...]` override.
DEFAULT_VARIATIONS = ('ball.speed',)

DEFAULT_BALL_COLOR = (0.85, 0.12, 0.12)  # red


class CloudgripperMuJoCoBall(CustomMuJoCoEnv):
    """A ball crossing an empty CloudGripper cell left->right at constant speed.

    Deliberately does not extend CloudgripperMuJoCoEnv: that class verifies the
    arm's bodies/joints/actuators exist at compile time and calibrates
    actuation ranges from them, none of which applies to an arm-free scene.

    Attributes:
        z_floor_top: top surface of the ground plate — half its thickness,
            the same constant the cube task spawns objects on (the plate
            spans +-0.0015 in z, +-0.125 in y and +-0.16 in x).
        ball_radius: sphere radius; the ball is centred one radius above
            z_floor_top so it looks like it is resting on the plate.
        travel_range: (start, end) along the travel axis. The episode ends on
            the first step at or past `end`, which overshoots it by up to
            speed_range[1] * control_timestep (1 cm); `end` is set so even
            that lands within the plate's +-0.125. Stopping on the overshoot
            rather than clamping to `end` keeps every step's displacement
            exactly equal.
        ball_cross_pos: position on the axis across the travel direction —
            0.0 puts the ball down the middle of the scene.
        speed_range: bounds of the per-episode speed in m/s — the default is
            the in-distribution band; pass a disjoint one to collect
            out-of-distribution evaluation sets. Pick it against the *model
            input* resolution S (what the training transform resizes to, not
            what is stored): per-frame displacement is 0.139 * speed * S px,
            and the ball is 0.067 * S px across. Below ~1.5 px/frame the
            motion is lost in compression noise; above ~0.29 m/s consecutive
            frames stop overlapping the ball at any resolution, since both
            scale with S. See crossing_steps() for how speed sets episode
            length.
    """

    folder_path = Path(__file__).resolve().parent
    xml_file = "cloudgripper_static_scene.xml"

    main_camera = "Camera_main"

    z_floor_top = 0.0015

    ball_radius = 0.012
    travel_range = (-0.115, 0.115)
    ball_cross_pos = 0.0
    default_speed_range = (0.045, 0.20)

    metadata: dict = {
        "render_modes": ["rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        ob_type: str = "pixels",
        multiview: bool = False,
        height: int = 224,
        width: int = 224,
        speed_range: tuple[float, float] | None = None,
        physics_timestep: float = 0.002,
        control_timestep: float = 0.05,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
            height=height,
            width=width,
            **kwargs,
        )

        self._obt_type: str = ob_type
        self._multivew: bool = multiview

        # tuple(): Hydra hands this over as a ListConfig
        self.speed_range = tuple(
            self.default_speed_range if speed_range is None else speed_range
        )
        if not self.speed_range[0] <= self.speed_range[1]:
            raise ValueError(f'speed_range must be (low, high), got {self.speed_range}')

        self._steps: int = 0
        self._ball_speed: float = float(np.mean(self.speed_range))
        self._travel_pos: float = self.travel_range[0]
        self._ball_pos: np.ndarray = self._ball_world_pos(self.travel_range[0])

        self.variation_space = swm_spaces.Dict({
            'ball': swm_spaces.Dict({
                'color': swm_spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(3,),
                    dtype=np.float64,
                    init_value=np.array(DEFAULT_BALL_COLOR),
                ),
                'speed': swm_spaces.Box(
                    low=self.speed_range[0],
                    high=self.speed_range[1],
                    shape=(1,),
                    dtype=np.float64,
                    init_value=np.array([float(np.mean(self.speed_range))]),
                ),
            }),
        })

    @classmethod
    def crossing_steps(cls, speed: float, control_timestep: float = 0.05) -> int:
        """Control steps a full crossing takes at `speed`.

        Slow speeds make long episodes: an episode capped below this (by
        max_episode_steps) is cut off mid-sweep, which silently breaks the
        "every episode is one complete crossing" property. Collection scripts
        use this to check the cap before collecting rather than after.

        args:
            speed: ball speed in m/s
            control_timestep: seconds per control step
        returns:
            number of steps to cross the full travel range
        """
        span = cls.travel_range[1] - cls.travel_range[0]
        return int(np.ceil(span / (speed * control_timestep)))

    def build_mjcf_model(self) -> mjcf.RootElement:
        """Loads the arm-free cell scene."""
        return mjcf.from_path(str(self.folder_path / self.xml_file))

    def modify_mjcf_model(self, mjcf_model):
        """Spawns the ball as a non-colliding mocap body if not already present.

        args:
            mjcf_model: the scene to modify
        returns:
            modified mujoco scene
        """
        if mjcf_model.find('body', 'ball_0') is None:
            ball = mjcf_model.worldbody.add(
                'body',
                name='ball_0',
                mocap=True,
                pos=self._ball_world_pos(self.travel_range[0]).tolist(),
            )
            ball.add(
                'geom',
                name='ball_0',
                type='sphere',
                size=[self.ball_radius],
                rgba=[*DEFAULT_BALL_COLOR, 1.0],
                contype=0,
                conaffinity=0,
                group=1,
            )

        return mjcf_model

    def post_compilation(self) -> None:
        """Caches the ball's geom and mocap ids after the model compiles."""
        self._ball_geom_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_GEOM,
            'ball_0',
        )
        body_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_BODY,
            'ball_0',
        )
        self._ball_mocap_id = int(self._model.body_mocapid[body_id])

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
        *args,
        **kwargs,
    ) -> tuple[dict, dict]:
        """Resets environment to initial state. Also performs variation of scene."""
        options = options or {}

        swm_spaces.reset_variation_space(
            self.variation_space,
            seed=seed,
            options=options,
            default_variations=DEFAULT_VARIATIONS,
        )

        return super().reset(seed=seed, options=options, *args, **kwargs)

    def initialize_episode(self) -> None:
        """Starts a new crossing: sample this episode's speed (and colour, if
        varied) and place the ball at the start of its travel range."""
        self._steps = 0
        self._ball_speed = float(self.variation_space['ball']['speed'].value[0])

        color = self.variation_space['ball']['color'].value
        self._model.geom(self._ball_geom_id).rgba[:3] = color

        self._set_ball_pos(self.travel_range[0])
        mujoco.mj_forward(self._model, self._data)

    def _ball_world_pos(self, travel_pos: float) -> np.ndarray:
        """World position of the ball's centre at `travel_pos` along its path.

        The ball travels along +y, which is rightwards in the main camera's
        image, and is centred one radius above the plate so it looks like it
        rests on it.

        args:
            travel_pos: position along the travel axis
        returns:
            [x, y, z] world position of the ball centre
        """
        return np.array(
            [
                self.ball_cross_pos,
                travel_pos,
                self.z_floor_top + self.ball_radius,
            ],
            dtype=np.float64,
        )

    def _set_ball_pos(self, travel_pos: float) -> None:
        """Moves the mocap ball to `travel_pos` along its path.

        args:
            travel_pos: position along the travel axis
        """
        self._travel_pos = travel_pos
        self._ball_pos = self._ball_world_pos(travel_pos)
        self._data.mocap_pos[self._ball_mocap_id] = self._ball_pos

    def pre_step(self) -> None:
        """Advances the ball by one control step at this episode's speed.

        Set before mj_step so the step's own kinematics pass picks the new
        position up — no extra mj_forward needed.
        """
        self._steps += 1
        self._set_ball_pos(
            self.travel_range[0]
            + self._ball_speed * self._steps * self._control_timestep
        )

    def set_control(self, action: np.ndarray) -> None:
        """Ignores the action — nothing in this scene is actuated.

        args:
            action: ignored
        """
        pass

    def truncate_episode(self) -> bool:
        """Ends the episode once the ball has crossed the scene, so every
        episode is exactly one left->right sweep rather than a sweep followed
        by the ball sitting off-screen.
        """
        return self._travel_pos >= self.travel_range[1]

    def compute_observation(self) -> dict:
        """Placeholder state: the scene has no controllable degrees of freedom,
        and the ball's true position is deliberately withheld here (it is in
        the info dict instead) so it can serve as an unseen probe target.
        """
        return {"state": np.zeros(5, dtype=np.float32)}

    def compute_reward(self) -> float:
        """No task objective — this env exists to generate observations."""
        return 0.0

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {"state": Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)}
        )

    @property
    def action_space(self):
        """Inert, but kept 5-dimensional so datasets from this env share a
        schema with the robot tasks and train with the same config."""
        return swm_spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )

    def _ball_info(self) -> dict:
        """Ground-truth generating factors, stored as dataset columns."""
        return {
            "ball_pos": self._ball_pos.astype(np.float32),
            "ball_speed": np.array([self._ball_speed], dtype=np.float32),
            "ball_color": self.variation_space['ball']['color'].value.astype(
                np.float32
            ),
        }

    def get_reset_info(self) -> dict:
        info = super().get_reset_info()
        info.update(self._ball_info())
        return info

    def get_step_info(self) -> dict:
        info = super().get_step_info()
        info.update(self._ball_info())
        return info

    def render(self, camera=None, *args, **kwargs):
        """Renders the cell's main camera by default."""
        camera = camera or self.main_camera
        return super().render(camera=camera, *args, **kwargs)

    def close(self):
        """Releases the offscreen renderer's GL context."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    env = CloudgripperMuJoCoBall(height=854, width=480)
    try:
        obs, info = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        frame = env.render()
        print(obs, reward, info["ball_pos"], info["ball_speed"])
        from PIL import Image
        img = Image.fromarray(frame)
        img.show()

    finally:
        env.close()
