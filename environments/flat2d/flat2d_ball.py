"""Flat 2D ball scene — a ball crossing a plain wall on a plain floor.

The simplest possible sibling of ``cloudgripper_mj/cloudgripper_mj_ball.py``:
a ball travelling left->right at a speed drawn per episode, with nothing else
in the frame. Where the MuJoCo version keeps a photorealistic robot cell in
the background, this one keeps nothing — the image is a wall-coloured
background, a floor-coloured band across the bottom, and a ball-coloured
circle resting on it. Three regions, three colours, no texture, no shading,
no shadow.

That is the point. A rendered frame here contains **exactly three distinct RGB
triples**, so the pixel-space content of the dataset is fully described by
(wall colour, floor colour, ball colour, ball position, ball radius). A latent
probed against those factors has nothing else it could be encoding.

Why not MuJoCo: MuJoCo always shades geometry against a light, antialiases
silhouettes and tessellates a sphere into triangles, so "the ball" would come
out as a spread of hundreds of near-red values with a gradient across it. No
material setting removes all three effects. So the scene is built in **pymunk**
(2D rigid-body physics) and rasterised directly with numpy — a hard-edged
circle test per pixel, no antialiasing and no interpolation, which is what
makes the three-colour guarantee exact rather than approximate.

Why pymunk rather than nothing at all: the space owns the scene geometry
(floor extent, ball radius and position) and integrates the ball's motion, so
the renderer reads the same geometry the physics holds instead of a duplicate
set of constants.

The ball is a **kinematic** body, which is pymunk's exact analogue of the
MuJoCo version's mocap body: kinematic bodies are integrated straight from
their velocity and are not touched by gravity, contacts or friction. The
episode's sampled speed is therefore the speed for the whole episode, with
nothing able to perturb it.

Ground-truth factors land in the step info (and hence as dataset columns) as
``ball_pos``/``ball_speed``/``ball_color``/``floor_color``/``wall_color``, for
probing a learned latent against what actually generated the frames. They are
deliberately kept out of the ``state`` observation so a model conditioned on
state can't read the answer off its input.

Coordinates
-----------
World units are fractions of the **image width**: x runs 0 (left edge) to 1
(right edge), y runs 0 (bottom edge) upward, so a square image spans
y in [0, 1] and a wider one spans y in [0, height/width]. Both axes use the
same scale, so a circle in world units is a circle in pixels at any
resolution, and nothing about the scene changes when the render size does.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pymunk
from gymnasium.spaces import Box
from stable_worldmodel import spaces as swm_spaces


# Only the speed is resampled per episode; every colour stays at its
# init_value, so one dataset really does contain only three RGB triples.
# Pass e.g. variation=[ball.speed,ball.color] (via the collection script) to
# resample a colour too.
DEFAULT_VARIATIONS = ('ball.speed',)

DEFAULT_WALL_COLOR = (255, 255, 255)  # white background
DEFAULT_FLOOR_COLOR = (0, 0, 0)  # black floor band
DEFAULT_BALL_COLOR = (255, 0, 0)  # red ball


class Flat2DBall(gym.Env):
    """A ball crossing a flat two-colour 2D scene left->right at constant speed.

    Attributes:
        floor_height: thickness of the floor band, in world units (fractions
            of image width) measured up from the bottom edge. The band always
            spans the full image width.
        ball_radius: ball radius in world units. The ball is centred exactly
            one radius above the floor's top edge, so it rests on it.
        travel_range: (start, end) x positions of the ball centre. Set so the
            ball is fully inside the frame at both ends. The episode ends on
            the first step at or past `end`, which overshoots it by up to
            speed_range[1] * control_timestep; `end` leaves room for that.
        speed_range: bounds of the per-episode speed, in image widths per
            second — the default is the in-distribution band; pass a disjoint
            one to collect out-of-distribution evaluation sets. Resolution
            independent: at model input size S px, per-frame displacement is
            `speed * control_timestep * S` px and the ball is
            `2 * ball_radius * S` px across, so both ends of the band scale
            together. At the defaults that is 0.01-0.04 image widths per
            frame against a ball 0.07 widths across — always moving visibly,
            never jumping further than its own diameter. See crossing_steps()
            for how speed sets episode length.
    """

    floor_height = 0.25
    ball_radius = 0.035
    travel_range = (0.08, 0.92)
    default_speed_range = (0.2, 0.8)

    metadata: dict = {
        "render_modes": ["rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        ob_type: str = "pixels",
        height: int = 224,
        width: int = 224,
        speed_range: tuple[float, float] | None = None,
        control_timestep: float = 0.05,
        wall_color: tuple[int, int, int] | None = None,
        floor_color: tuple[int, int, int] | None = None,
        ball_color: tuple[int, int, int] | None = None,
        ball_colors: list | None = None,
        ball_radius: float | None = None,
        zero_action: bool = True,
        render_mode: str | None = None,
        **kwargs,
    ):
        """Builds the pymunk scene and the pixel grid used to rasterise it.

        The three colour arguments set each element's *fixed* colour — the
        init_value its variation entry is reset to at the start of every
        episode. Changing one does not touch the variation RNG stream, so two
        collection runs that differ only in a colour produce the same speeds,
        the same ball positions and the same episode lengths frame for frame.
        (Listing a colour in `variation` instead makes it a sampled factor,
        which does consume draws and would desynchronise two such runs.)

        args:
            ob_type: kept for signature parity with the MuJoCo ball env; the
                observation is a placeholder either way (see
                compute_observation).
            height: rendered image height in pixels
            width: rendered image width in pixels
            speed_range: (low, high) episode speed in image widths per second
            control_timestep: seconds advanced per env step
            wall_color: fixed background RGB in 0-255
            floor_color: fixed floor-band RGB in 0-255
            ball_color: fixed ball RGB in 0-255
            ball_radius: ball radius in world units, overriding the class
                default. Must leave the ball fully inside the frame across
                `travel_range`, which is checked.
            zero_action: collapse the action space to the single point 0, so
                the recorded action column is constant rather than noise. See
                the action_space property for why that is the default.
            render_mode: unused; accepted for gymnasium compatibility
        """
        super().__init__()

        self._ob_type: str = ob_type
        self.render_mode = render_mode
        self._render_height = int(height)
        self._render_width = int(width)
        self._control_timestep = float(control_timestep)

        if ball_radius is not None:
            self.ball_radius = float(ball_radius)
        lo, hi = self.travel_range
        if lo - self.ball_radius <= 0 or hi + self.ball_radius >= 1:
            raise ValueError(
                f'ball_radius={self.ball_radius} puts the ball outside the frame '
                f'over travel_range={self.travel_range}: it would span '
                f'{lo - self.ball_radius:.4f} to {hi + self.ball_radius:.4f}, '
                f'and the frame is 0 to 1.'
            )

        wall_rgb = self._as_rgb_tuple(wall_color, DEFAULT_WALL_COLOR, 'wall_color')
        floor_rgb = self._as_rgb_tuple(floor_color, DEFAULT_FLOOR_COLOR, 'floor_color')
        ball_rgb = self._as_rgb_tuple(ball_color, DEFAULT_BALL_COLOR, 'ball_color')

        # tuple(): Hydra hands this over as a ListConfig
        self.speed_range = tuple(
            self.default_speed_range if speed_range is None else speed_range
        )
        if not self.speed_range[0] <= self.speed_range[1]:
            raise ValueError(f'speed_range must be (low, high), got {self.speed_range}')

        self._build_space()
        self._build_pixel_grid()

        self._steps: int = 0
        self._ball_speed: float = float(np.mean(self.speed_range))
        self._wall_color = np.array(wall_rgb, dtype=np.uint8)
        self._floor_color = np.array(floor_rgb, dtype=np.uint8)
        self._ball_color = np.array(ball_rgb, dtype=np.uint8)

        # Built once and reused, not rebuilt per access: seeding an action
        # space only sticks if the caller keeps getting the same object back
        # (swm's RandomPolicy.set_seed and EverythingToInfoWrapper's reset-time
        # sample both go through this property), and an unseeded action space
        # would make the recorded action column differ between two otherwise
        # identical runs.
        self._action_space = swm_spaces.Box(
            low=0.0,
            high=0.0 if zero_action else 1.0,
            shape=(5,),
            dtype=np.float32,
        )

        self._ball_color_choices = (
            None if ball_colors is None
            else [self._to_rgb(np.asarray(c, dtype=np.float64)) for c in ball_colors]
        )
        if self._ball_color_choices is not None and len(self._ball_color_choices) == 0:
            raise ValueError('ball_colors was given but empty')

        self.variation_space = swm_spaces.Dict({
            'wall': swm_spaces.Dict({
                'color': self._color_space(wall_rgb),
            }),
            'floor': swm_spaces.Dict({
                'color': self._color_space(floor_rgb),
            }),
            'ball': swm_spaces.Dict({
                'color': self._color_space(ball_rgb),
                'speed': swm_spaces.Box(
                    low=self.speed_range[0],
                    high=self.speed_range[1],
                    shape=(1,),
                    dtype=np.float64,
                    init_value=np.array([float(np.mean(self.speed_range))]),
                ),
            }),
        })

    @staticmethod
    def _as_rgb_tuple(
        value, default: tuple[int, int, int], name: str
    ) -> tuple[int, int, int]:
        """Normalises a colour argument to a validated integer RGB triple.

        args:
            value: an RGB triple, or None to take `default`. Hydra hands these
                over as a ListConfig, hence the list() round-trip.
            default: colour to use when `value` is None
            name: argument name, for the error message
        returns:
            (r, g, b) ints in 0-255
        """
        if value is None:
            return tuple(default)
        rgb = tuple(int(c) for c in list(value))
        if len(rgb) != 3 or not all(0 <= c <= 255 for c in rgb):
            raise ValueError(
                f'{name} must be three integers in 0-255, got {tuple(value)}'
            )
        return rgb

    @staticmethod
    def _color_space(init_value: tuple[int, int, int]) -> swm_spaces.Box:
        """A 0-255 RGB variation entry.

        Colours are held in 0-255 rather than 0-1 so the values in the config,
        in the variation space and in the dataset columns are the same numbers
        as the pixels they produce. They are rounded to integers at render
        time, so a sampled colour is still exactly one RGB triple.

        args:
            init_value: the RGB triple this entry holds unless resampled
        returns:
            a variation-space Box over RGB
        """
        return swm_spaces.Box(
            low=0.0,
            high=255.0,
            shape=(3,),
            dtype=np.float64,
            init_value=np.array(init_value, dtype=np.float64),
        )

    @classmethod
    def crossing_steps(cls, speed: float, control_timestep: float = 0.05) -> int:
        """Control steps a full crossing takes at `speed`.

        Slow speeds make long episodes: an episode capped below this (by
        max_episode_steps) is cut off mid-sweep, which silently breaks the
        "every episode is one complete crossing" property. Collection scripts
        use this to check the cap before collecting rather than after.

        args:
            speed: ball speed in image widths per second
            control_timestep: seconds per control step
        returns:
            number of steps to cross the full travel range
        """
        span = cls.travel_range[1] - cls.travel_range[0]
        return int(np.ceil(span / (speed * control_timestep)))

    # -- scene ------------------------------------------------------------

    def _build_space(self) -> None:
        """Creates the pymunk space: a static floor box and a kinematic ball.

        Gravity is zero and the ball is kinematic, so neither the floor nor
        gravity can alter the ball's motion — the floor is here because it is
        part of the scene's geometry (the renderer reads its extent) and
        because an object visibly resting on a surface should be resting on a
        surface in the model too.
        """
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 0.0)

        # The floor spans the full image width; `top` is what the ball rests on.
        self._floor_shape = pymunk.Poly(
            self._space.static_body,
            [
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, self.floor_height),
                (0.0, self.floor_height),
            ],
        )

        self._ball_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._ball_body.position = self._ball_start_pos()
        self._ball_shape = pymunk.Circle(self._ball_body, self.ball_radius)

        self._space.add(self._floor_shape, self._ball_body, self._ball_shape)

    def _ball_start_pos(self) -> tuple[float, float]:
        """Ball centre at the start of its travel, resting on the floor.

        returns:
            (x, y) in world units
        """
        return (
            float(self.travel_range[0]),
            float(self.floor_height + self.ball_radius),
        )

    @property
    def _ball_pos(self) -> np.ndarray:
        """Ball centre as [x, y] in world units, read back from the space."""
        pos = self._ball_body.position
        return np.array([pos.x, pos.y], dtype=np.float64)

    # -- rendering --------------------------------------------------------

    def _build_pixel_grid(self) -> None:
        """Precomputes pixel-centre coordinates for the circle test.

        Pixel centres sit at integer + 0.5, in the same world units as the
        space, with the y axis flipped because image row 0 is the top.
        """
        scale = float(self._render_width)  # world x in [0, 1] spans the width
        xs = (np.arange(self._render_width, dtype=np.float64) + 0.5) / scale
        ys = (
            self._render_height - (np.arange(self._render_height, dtype=np.float64) + 0.5)
        ) / scale
        self._px_x = xs[None, :]
        self._px_y = ys[:, None]

        # The floor is static, so its row mask never changes.
        self._floor_mask = (self._px_y <= self._floor_shape.bb.top) & (
            self._px_y >= self._floor_shape.bb.bottom
        )

    def render(self, *args, **kwargs) -> np.ndarray:
        """Rasterises the scene as a flat three-colour RGB image.

        Every pixel is assigned one of the three colours outright — no
        antialiasing, no blending, no lighting — so the returned frame
        contains at most three distinct RGB triples (exactly three whenever
        the ball is in frame and the colours differ).

        returns:
            (height, width, 3) uint8 RGB image
        """
        img = np.empty(
            (self._render_height, self._render_width, 3), dtype=np.uint8
        )
        img[:, :] = self._wall_color
        img[np.broadcast_to(self._floor_mask, img.shape[:2])] = self._floor_color

        cx, cy = self._ball_pos
        inside = (self._px_x - cx) ** 2 + (self._px_y - cy) ** 2 <= (
            self.ball_radius**2
        )
        img[inside] = self._ball_color

        return img

    # -- gymnasium API ----------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
        **kwargs,
    ) -> tuple[dict, dict]:
        """Resets the environment and resamples the scene's variations."""
        super().reset(seed=seed, options=options, **kwargs)
        options = options or {}

        swm_spaces.reset_variation_space(
            self.variation_space,
            seed=seed,
            options=options,
            default_variations=DEFAULT_VARIATIONS,
        )
        self.initialize_episode()

        return self.compute_observation(), self.get_reset_info()

    def initialize_episode(self) -> None:
        """Starts a new crossing: apply this episode's colours and speed, and
        put the ball back at the start of its travel range."""
        self._steps = 0

        self._wall_color = self._to_rgb(self.variation_space['wall']['color'].value)
        self._floor_color = self._to_rgb(self.variation_space['floor']['color'].value)
        self._ball_color = self._to_rgb(self.variation_space['ball']['color'].value)
        if self._ball_color_choices is not None:
            # A discrete palette overrides whatever the variation space holds.
            # The variation space's colour axis is a continuous Box, which can
            # only hold one colour fixed or sample the whole cube; neither gives
            # a dataset built from a named handful of colours. Drawn from the
            # env's own RNG so it follows `seed` like every other per-episode
            # draw.
            i = int(self.np_random.integers(len(self._ball_color_choices)))
            self._ball_color = self._ball_color_choices[i]

        self._ball_speed = float(self.variation_space['ball']['speed'].value[0])
        self._ball_body.position = self._ball_start_pos()
        self._ball_body.velocity = (self._ball_speed, 0.0)

    @staticmethod
    def _to_rgb(value: np.ndarray) -> np.ndarray:
        """Rounds a sampled colour to the integer RGB triple actually drawn.

        args:
            value: RGB in 0-255, possibly fractional
        returns:
            uint8 RGB triple
        """
        return np.rint(np.asarray(value)).clip(0, 255).astype(np.uint8)

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Advances the ball by one control step at this episode's speed.

        A single pymunk step covers the whole control timestep: the only
        moving body is kinematic, so its integration is exact and there are no
        contacts to resolve in substeps.

        args:
            action: ignored — nothing in this scene is actuated
        returns:
            the standard gymnasium (obs, reward, terminated, truncated, info)
        """
        self._steps += 1
        self._space.step(self._control_timestep)

        return (
            self.compute_observation(),
            self.compute_reward(),
            self.terminate_episode(),
            self.truncate_episode(),
            self.get_step_info(),
        )

    def terminate_episode(self) -> bool:
        """No failure or success condition — the scene just plays out."""
        return False

    def truncate_episode(self) -> bool:
        """Ends the episode once the ball has crossed the scene, so every
        episode is exactly one left->right sweep rather than a sweep followed
        by the ball sitting off-screen.
        """
        return bool(self._ball_body.position.x >= self.travel_range[1])

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
        schema with the robot tasks and train with the same config.

        Collapsed to the single point 0 unless `zero_action=False`. Nothing in
        this scene is actuated, so a wider space would have every sampled
        action be pure noise in the dataset's action column — and downstream
        that noise is not harmlessly ignored: LeWM feeds the action embedding
        to AdaLN-zero conditioning, where it is the only input driving the
        shift/scale/gate of every predictor block, so the model has to spend
        capacity learning that a random input should produce a constant
        modulation. A constant column z-scores to zeros and removes that.

        Note this covers the whole action column, not just the policy's
        output: swm's EverythingToInfoWrapper samples this space directly for
        the reset-step action, which no choice of policy can override.
        """
        return self._action_space

    def _ball_info(self) -> dict:
        """Ground-truth generating factors, stored as dataset columns.

        Colours are the rounded values actually drawn, not the raw samples, so
        a probe target always matches the pixels exactly.
        """
        return {
            "ball_pos": self._ball_pos.astype(np.float32),
            "ball_speed": np.array([self._ball_speed], dtype=np.float32),
            "ball_color": self._ball_color.astype(np.float32),
            "floor_color": self._floor_color.astype(np.float32),
            "wall_color": self._wall_color.astype(np.float32),
        }

    def get_reset_info(self) -> dict:
        return dict(self._ball_info())

    def get_step_info(self) -> dict:
        return dict(self._ball_info())

    def close(self):
        """Drops the pymunk space.

        The renderer is pure numpy and holds nothing, but the space wraps a
        Chipmunk object freed through a cffi ``ffi.gc`` callback. Left alive
        until interpreter shutdown, that callback runs against a half-torn-down
        pymunk module and prints an ignored ``AttributeError``; dropping the
        reference here lets it be collected while the interpreter is still up.
        """
        self._space = None
        self._floor_shape = None
        self._ball_shape = None
        self._ball_body = None


if __name__ == "__main__":
    env = Flat2DBall(height=448, width=448)
    try:
        obs, info = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )
        frame = env.render()
        print(obs, reward, info["ball_pos"], info["ball_speed"])
        print(
            'unique colours in frame:',
            np.unique(frame.reshape(-1, 3), axis=0),
        )
        from PIL import Image
        img = Image.fromarray(frame)
        img.show()

    finally:
        env.close()
