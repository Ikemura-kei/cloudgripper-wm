from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from dm_control import mjcf
from gymnasium.spaces import Box
from environments.cloudgripper_mj.cloudgripper_mj_env import CloudgripperMuJoCoEnv
from stable_worldmodel import spaces as swm_spaces


DEFAULT_VARIATIONS = (
    'agent.start_pos',
    'agent.goal_pos',
)


class CloudgripperMuJoCoTracking(CloudgripperMuJoCoEnv):
    """Data collection for arbitrary CloudGripper motion."""

    def __init__(
        self,
        ob_type: str ="pixels",
        multiview: bool =False,
        height: int =224,
        width: int =224,
        *args,
        **kwargs,
    ):
        super().__init__(*args, height=height, width=width, **kwargs)

        self._obt_type = ob_type
        self._multivew = multiview
        self._goal_pos: np.ndarray | None = None
        self._goal_image: np.ndarray | None = None


        self.variation_space = swm_spaces.Dict({
            'agent' : swm_spaces.Dict({
                'start_pos': swm_spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(5,),
                    dtype=np.float64,
                    init_value=self.initial_pose,
                ),
                'goal_pos' : swm_spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(5,),
                    dtype=np.float64,
                    init_value=self.initial_pose,
                )
            }),
            'material': swm_spaces.Dict({
                'color': swm_spaces.Dict({
                    key: swm_spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(3,),
                        dtype=np.float64,
                        init_value=val.copy(),
                    )
                    for key, val in self.material_colors.items()
                })
            }),
            'light': swm_spaces.Dict({
                'intensity': swm_spaces.Box(
                    low=0.0,
                    high=2.0,
                    shape=(1,),
                    dtype=np.float64,
                    init_value=np.array([self.light_intensity]),
                ),
            }),
        })

    def modify_mjcf_model(self, mjcf_model):
        """
        Modify any mjcf models (cloudgripper model). Returns updated model
        
        During reset, the modifications are followed by rebuild of mujoco model.
        """
        # Add anything here that requires rebuild of mujoco model.
        return mjcf_model

    def reset(
        self,
        seed: int | None = None,
        options: dict | None =None,
        *args,
        **kwargs,
    ) -> tuple[dict, dict]:
        """Resets environment to initial space. Also performs variation of scene."""
        options = options or {}

        swm_spaces.reset_variation_space(
            self.variation_space,
            seed=seed,
            options=options,
            default_variations=DEFAULT_VARIATIONS,
        )

        obs, info = super().reset(seed=seed, options=options, *args, **kwargs)

        return obs, info


    def initialize_episode(self) -> None:
        """
        Initializes new episode applying variations to the scene
            1) random start position of gripper (DEFAULT_VARIATION)
            2) random goal position of gripper (DEFAULT_VARIATION)
            3) random color tint for every material in cloudgripper_scene.xml
            4) random color/intensity for the LED light strips
        """

        # Sample new random configuration. 
        self._target_pos = self.variation_space['agent']['start_pos'].value.astype(np.float32)
        self._current_pos = self._target_pos.copy()
        self.set_active_joints(self._current_pos)

        # In task mode, sample a goal and goal image
        if self._mode == "task":
            self._goal_pos = self.variation_space['agent']['goal_pos'].value.astype(np.float32)
            self.set_active_joints(self._goal_pos)
            self._goal_image = self.render()
            self.set_active_joints(self._target_pos)  # restore to start_pos
        else:
            self._goal_pos = None
            self._goal_image = None

        # Vary material colors
        for name in self.material_colors.keys():
            color = self.variation_space['material']['color'][name].value
            self._model.material(name).rgba[:3] = color

        # Vary intensity of light sources
        intensity = self.variation_space['light']['intensity'].value[0]
        self._model.material('led_light').emission = intensity

        led_color = self.variation_space['material']['color']['led_light'].value
        for name in self.lightbulb_names:
            self._model.light(name).diffuse[:] = led_color * intensity

    def compute_reward(self):
        return 0.0

    def get_reset_info(self) -> dict:
        info = super().get_reset_info()
        if self._goal_image is not None:
            info["goal"] = self._goal_image
        return info

    def get_step_info(self) -> dict:
        info = super().get_step_info()
        if self._goal_image is not None:
            info["goal"] = self._goal_image
        return info


if __name__ == "__main__":
    env = CloudgripperMuJoCoTracking(height=600,  width=600)
    try:
        obs, info = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        frame = env.render()
        print(obs, reward)
        from PIL import Image
        img = Image.fromarray(frame)
        img.show()

    finally:
        env.close()