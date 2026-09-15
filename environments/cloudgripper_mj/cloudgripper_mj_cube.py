"""CloudGripper MuJoCo cube task — ported from ogbench.manipspace.envs.cube_env.CubeEnv.

ogbench's CubeEnv is built on ManipSpaceEnv (a full 6-DoF arm with mocap-based
IK targets, oracle policies, per-task init/goal position tables for up to 8
cubes). CloudgripperMuJoCoEnv is a much simpler 5-DoF position-controlled rig
(x/y/z/rotate/gripper via CloudgripperMuJoCoEnv.set_control()) built directly
on the lower-level CustomMuJoCoEnv, not ManipSpaceEnv — so none of
ManipSpaceEnv's arm/oracle machinery (self._arm_joint_ids, initialize_arm(),
compute_ob_info(), lie.SO3 helpers, etc.) is available here. This file ports
the *concept* (a free-jointed cube, reset randomization, distance-based
success) using CloudgripperMuJoCoEnv's own control idiom, scoped down to a
single cube (ogbench's `env_type='single'`) rather than the
double/triple/quadruple/octuple stacking variants.

Cube size is varied the same way stable_worldmodel's own ogbench cube env
does it (see third_party/stable-worldmodel/stable_worldmodel/envs/ogbench/
cube_env.py's modify_mjcf_model): a fresh continuous sample every episode,
written onto the mesh asset's `scale` in the mjcf tree, recompiling the
model (mark_dirty()) whenever it actually changes. Measured on this scene:
a full recompile costs ~114ms vs ~0.035ms for the ordinary reset path — a
real but modest per-episode cost (roughly 3-6% of a typical episode's
wall-clock time here), traded for genuinely continuous size coverage and
much simpler code than pre-baking discrete size variants.

Several numeric choices below are ASSUMPTIONS made without hardware/visual
confirmation — search for "ASSUMPTION" and adjust as needed. Everything
tagged "measured" was derived directly from cloudgripper_scene.xml via
mujoco raycasting/forward-kinematics, not guessed.
"""

import mujoco
import numpy as np
from dm_control import mjcf
from environments.cloudgripper_mj.cloudgripper_mj_env import CloudgripperMuJoCoEnv
from stable_worldmodel import spaces as swm_spaces


DEFAULT_VARIATIONS = (
    'agent.start_pos',
    'agent.goal_pos',
    'cube',
    'material',
    'light',
)

class CloudgripperMuJoCoCube(CloudgripperMuJoCoEnv):
    """Single-cube pick-and-place task with CloudGripper.
    
        z_spawn_cube: equals half of cloudgripper's ground plate thickness
        cube_size_range: measured with respect to gripper-opening length
        success_dist_threshold: target cube position error
        success_yaw_threshold: target cube rotation error
        expert_phases: describes the state-machine of the pick-and-place task
    """

    z_spawn_cube = 0.0015  # equal to half of cloudgripper's ground-plate thickness
    cube_size_range = (0.005, 0.025)
    success_dist_threshold = 0.004
    success_yaw_threshold = np.pi/12 # 15 degrees
    expert_phases = [
        "approach",
        "open",
        "descend",
        "grasp",
        "lift",
        "transit",
        "place",
        "release",
        "elevate",
        "retreat",
        "done",
    ]

    def __init__(
        self,
        ob_type: str = "pixels",
        multiview: bool = False,
        height: int = 224,
        width: int = 224,
        ws_xydim: tuple[int, int] = (0.16, 0.12),
        *args,
        **kwargs,
    ):
        super().__init__(*args, height=height, width=width, **kwargs)

        self._obt_type: str = ob_type
        self._multivew: bool = multiview
        self._goal_image: np.ndarray | None = None
        self._success: bool = False
        self._ws_x = (-ws_xydim[0]/2, ws_xydim[0]/2) 
        self._ws_y = (-ws_xydim[1]/2, ws_xydim[1]/2)

        self.variation_space = swm_spaces.Dict({
            'agent': swm_spaces.Dict({
                'start_pos': swm_spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(5,),
                    dtype=np.float64,
                    init_value=self.initial_pose,
                ),
                'goal_pos': swm_spaces.Box(
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
            'cube': swm_spaces.Dict({
                'color': swm_spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(3,),
                    dtype=np.float64,
                    init_value=(0.96, 0.26, 0.33),
                ),
                'size': swm_spaces.Box(
                    low=self.cube_size_range[0],
                    high=self.cube_size_range[1],
                    shape=(1,),
                    dtype=np.float64,
                    init_value=np.array([0.03]),
                ),
                'start_pos': swm_spaces.Box(
                    low=np.array([0.0, 0.0]),
                    high=np.array([1.0, 1.0]),
                    shape=(2,),
                    dtype=np.float64,
                    init_value=(0.0, 0.0),
                ),
                'start_yaw': swm_spaces.Box(
                    low=0.0,
                    high=2 * np.pi,
                    shape=(1,),
                    dtype=np.float64,
                    init_value=np.array([0.0]),
                ),
                'goal_pos': swm_spaces.Box(
                    low=np.array([0.0, 0.0]),
                    high=np.array([1.0, 1.0]),
                    shape=(2,),
                    dtype=np.float64,
                    init_value=(0.02, 0.02),
                ),
                'goal_yaw': swm_spaces.Box(
                    low=0.0,
                    high=2 * np.pi,
                    shape=(1,),
                    dtype=np.float64,
                    init_value=np.array([0.0]),
                ),
            })
        })

    def modify_mjcf_model(self, mjcf_model):
        """Spawn the cube if it is not found in the scene. If the size has
        changed mark the environment as dirty for recompilation (as in OG-Bench).
        
        args:
            mjcf_model: the scene to modify
        returns
            modified mujoco scene
        """

        if mjcf_model.find('body', 'object_0') is None:
            assets_folder = self.folder_path / "assets"

            cube_class = mjcf.from_path(str(assets_folder / 'cube_class.xml'))
            mjcf_model.include_copy(cube_class)

            cube_object = mjcf.from_path(str(assets_folder / 'cube_object.xml'))
            mjcf_model.include_copy(cube_object)

        # cube.stl has side length 30mm -> division by 30 allows meter scaling
        self._cube_size = self.variation_space['cube']['size'].value[0]
        desired_scale = float(self._cube_size) / 30.0

        cube_mesh = mjcf_model.find('mesh', 'cube')
        size_changed = cube_mesh.scale is None or not np.allclose(
            cube_mesh.scale, desired_scale
        )
        cube_mesh.scale = [desired_scale, desired_scale, desired_scale]
        if size_changed:
            self.mark_dirty()

        return mjcf_model

    def post_compilation_objects(self) -> None:
        """Caches the cube geom id after the mujoco model compiles."""
        self._cube_geom_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_GEOM,
            'object_0'
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
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

        obs, info = super().reset(
            seed=seed,
            options=options,
            *args,
            **kwargs
        )

        return obs, info

    def initialize_episode(self) -> None:
        """Initializes a new episode for picking the cube.

         1) Sample a new goal pose for the cube and, if task mode is enabled,
            generate a goal image by moving the robot and cube to their goals.
         2) Sample a random start robot and start cube pose.
         3) Vary colors of the robot, the cube, and LED color and brightness.
         4) Calculate the amount of environment steps needed to complete the
            first phase of the expert trajectory (approaching the cube)
        """
        self._success = False
        self._goal_image = None

        # Cube goal pose
        cgx, cgy = self.variation_space['cube']['goal_pos'].value
        cube_goal_xy = (self.unnormalize(cgx, self._ws_x), self.unnormalize(cgy, self._ws_y))
        cube_goal_yaw = self.variation_space['cube']['goal_yaw'].value[0]
        self._cube_goal_xyz = np.array([*cube_goal_xy, self.z_spawn_cube], dtype=np.float32)
        self._cube_goal_yaw = float(cube_goal_yaw)
    
        self._target_pos = self.variation_space['agent']['start_pos'].value
        self._goal_pos = self.variation_space['agent']['goal_pos'].value

        # Task mode generates a goal image
        if self._mode == "task":
            self.set_active_joints(self._goal_pos)

            # generate goal image
            self._set_cube_pose(*cube_goal_xy, cube_goal_yaw)
            mujoco.mj_forward(self._model, self._data)
            self._goal_image = self.render()

        # Robot start pose
        self._current_pos = self._target_pos.copy()
        self.set_active_joints(self._current_pos)

        # Cube start pose
        csx, csy = self.variation_space['cube']['start_pos'].value
        cube_start_xy = (self.unnormalize(csx, self._ws_x), self.unnormalize(csy, self._ws_y))
        cube_start_yaw = self.variation_space['cube']['start_yaw'].value[0]
        self._set_cube_pose(*cube_start_xy, cube_start_yaw)

        # Cube color
        cube_col = self.variation_space['cube']['color'].value
        self._model.geom(self._cube_geom_id).rgba[:3] = cube_col

        # Scene material color
        for name in self.material_colors.keys():
            color = self.variation_space['material']['color'][name].value
            self._model.material(name).rgba[:3] = color

        # LED stripts intensity & color
        intensity = self.variation_space['light']['intensity'].value[0]
        self._model.material('led_light').emission = intensity

        led_color = self.variation_space['material']['color']['led_light'].value
        for name in self.lightbulb_names:
            self._model.light(name).diffuse[:] = led_color * intensity

        # update current simulation state
        mujoco.mj_forward(self._model, self._data)
        
        self._expert_phase_idx = 0
        first_waypoint = self._compute_expert_waypoint(
            self.expert_phases[self._expert_phase_idx],
        )
        self._expert_phase_steps_left = self._steps_for_move(first_waypoint)

    def _set_cube_pose(self, x: float, y: float, yaw: float):
        """Positions the cube at (x,y) with orientation yaw on the ground plate.
        
        args:
            x: x-position
            y: y-position
            yaw: z-orientation
        """
        cube_joint = self._data.joint('object_joint_0')
        cube_joint.qpos[:3] = (x, y, self.z_spawn_cube)
        cube_joint.qpos[3:] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        cube_joint.qvel[:] = 0.0


    def _steps_for_move(self, target: np.ndarray, settle=5) -> int:
        """Returns the number of environment steps required to reach a target
        configuration.
        
        args:
            target: the robot target configuration (5 joints)
            settle: additional number of steps for robot to settle
        returns:
            required number of steps + settle margin 
        """

        steps = 0
        for i, (j_name, a_name) in enumerate(zip(self.joint_names, self.actuator_names)):
            joint = self.model.joint(j_name)
            actuator = self.model.actuator(a_name)

            target_val = self.unnormalize(float(target[i]), joint.range)
            current_val = self.unnormalize(self._current_pos[i], joint.range)

            dist_mj = abs(target_val - current_val)
            v_max = actuator.ctrlrange[1]

            steps = max(steps, int(np.ceil(dist_mj / (v_max * self._control_timestep))))

        return steps + settle

    def _cube_xyz(self) -> float:
        """Get the current position of the cube object.
                
        returns:
            cube yaw in radians
        """
        return self._data.joint('object_joint_0').qpos[:3].copy()

    def _cube_yaw(self) -> float:
        """Get the current x-y orientation of the cube object.
        
        returns:
            cube yaw in radians
        """
        qw, _, _, qz = self._data.joint('object_joint_0').qpos[3:7]
        return 2.0 * float(np.arctan2(qz, qw))

    def _yaw_to_rot_norm(self, yaw: float) -> float:
        """rot_norm whose closing axis is aligned (mod 90deg) with the
        given world yaw — a cube has 4-fold rotational symmetry, so only
        yaw mod 90deg actually matters, and wrapping into (-pi/4, pi/4]
        keeps the result within the gripper's actual rotation range.
        """
        aligned_angle = ((yaw + np.pi / 4) % (np.pi / 2)) - np.pi / 4
        return aligned_angle / np.pi + 0.5

    def _compute_expert_waypoint(
        self,
        phase: str,
        rot_norm_override: float | None = None,
    ) -> np.ndarray:
        """Returns normalized [x, y, z, rot, grip] target for current phase.

         The gripper is open during the 'descend' and 'release' phases. It's
         orientation is aligned with the cube object during the pickup phase
         ('approach' -> 'descend' -> 'grasp') and aligns with the target yaw
         during the placing ('transit' -> 'place' -> 'release" -> 'retreat').
         If the 'done' phase is reached, the current pose is returned.

         args:
            phase: current phase name
            rot_norm_override: if given, use this rot_norm instead of the
                phase's default cube-aligned/goal-aligned value.
         returns:
            the arm's target pose in the current phase, current pose if done.
        """

        def ik(pose, rot_norm, grip):
            return self.solve_tcp_ik(pose, rot_norm, grip, self._current_pos[:3])

        transit_height = 0.15
        open = 0.0
        close = 1.0
        if phase in ("transit", "place", "release"):
            rot_norm = self._yaw_to_rot_norm(self._cube_goal_yaw)
            pos = self._cube_goal_xyz
        else:
            rot_norm = self._yaw_to_rot_norm(self._cube_yaw())
            cs_low, cs_up = self.cube_size_range
            off = (self._cube_size - cs_low)/(cs_up - cs_low)/2
            pos = self._cube_xyz() + np.array([0, 0, off]) # for large cubes lift grasp height

        if rot_norm_override is not None:
            rot_norm = rot_norm_override

        if phase == "approach":
            xy = ik(pos, rot_norm, close)[:2]
            return np.array([*xy, transit_height, rot_norm, close])
        elif phase== "open":
            xy = ik(pos, rot_norm, close)[:2]
            return np.array([*xy, transit_height, rot_norm, open])
        elif phase == "descend":
            xyz = ik(pos, rot_norm, open)
            return np.array([*xyz, rot_norm, open])
        elif phase == "grasp":
            xyz = ik(pos, rot_norm, close)
            return np.array([*xyz, rot_norm, close])
        elif phase == "lift":
            xy = ik(pos, rot_norm, close)[:2]
            return np.array([*xy, transit_height, rot_norm, close])
        elif phase == "transit":
            xy = ik(pos, rot_norm, close)[:2]
            return np.array([*xy, transit_height, rot_norm, close])
        elif phase == "place":
            xyz = ik(pos, rot_norm, close)
            return np.array([*xyz, rot_norm, close])
        elif phase == "release":
            xyz = ik(pos, rot_norm, open)
            return np.array([*xyz, rot_norm, open])
        elif phase == "elevate":
            xy = ik(pos, rot_norm, open)[:2]
            return np.array([*xy, transit_height, rot_norm, open])            
        elif phase == "retreat":
            xy = ik(self._goal_pos, rot_norm, close)[:2]
            return np.array([*xy, transit_height, rot_norm, close])
        else:
            return self._current_pos.copy()  # "done": hold position

    def _advance_expert_phase(self) -> np.ndarray:
        """Advance the phase state machine by one step.

         If the current phase's required environment steps have been
         taken the state machine advances to the next phase, unless
         the full trajectory is complete.

        returns:
            the current phase's normalized waypoint target.
        """

        phase = self.expert_phases[self._expert_phase_idx]
        if phase != "done":
            self._expert_phase_steps_left -= 1
            if (
                self._expert_phase_steps_left <= 0
                and self._expert_phase_idx < len(self.expert_phases) - 1
            ):
                self._expert_phase_idx += 1
                phase = self.expert_phases[self._expert_phase_idx]
                waypoint = self._compute_expert_waypoint(phase)
                self._expert_phase_steps_left = self._steps_for_move(waypoint)

                if phase == "transit":
                    self._transit_start_rot_norm = float(self._current_pos[3])
                    self._transit_end_rot_norm = self._yaw_to_rot_norm(self._cube_goal_yaw)
                    self._transit_total_steps = self._expert_phase_steps_left

        if phase == "transit":
            # linear interpolation from rotation during gripping to goal-aligned rotation
            total = max(self._transit_total_steps, 1)
            progress = float(np.clip(1.0 - self._expert_phase_steps_left / total, 0.0, 1.0))
            rot_norm = self._transit_start_rot_norm + progress * (
                self._transit_end_rot_norm - self._transit_start_rot_norm
            )
            return self._compute_expert_waypoint(phase, rot_norm_override=rot_norm).astype(np.float32)

        return self._compute_expert_waypoint(phase).astype(np.float32)


    def _compute_success(self) -> bool:
        """Returns true if the cube is close enough to the target.

        returns:
            true if positional and orientation error below thresholds.
        """
        cube_pos = self._cube_xyz()
        target_pos = self._cube_goal_xyz
        pos_err = np.linalg.norm(cube_pos - target_pos)

        cur_yaw = self._cube_yaw()
        goal_yaw = self._cube_goal_yaw
        yaw_err = abs(((cur_yaw - goal_yaw + np.pi/4) % (np.pi/2)) - np.pi/4)
        
        return (
            (pos_err <= self.success_dist_threshold) and
            (yaw_err <= self.success_yaw_threshold)
        )

    def compute_reward(self) -> float:
        """Determine the reward of the current state.
        If the cube has reached the goal, it turns green.

        returns:
            1 if success otherwise 0
        """
        self._success = self._compute_success()

        return 1.0 if self._success else 0.0

    def get_reset_info(self) -> dict:
        """Returns a dict with the current reset info.
        
         The dict contains:
            goal: the goal image in task_mode
            cube_pos: the current x-y-z position of the cube
            cube_yaw: the current rotation angle of the cube
            target_pos: the target pose of the cube 
            target_yaw: the target orientation of the cube 
            success: current state of the success condition
            expert_target: current targeted expert pose
         """
        info = super().get_reset_info()

        if self._goal_image is not None:
            info["goal"] = self._goal_image

        info["cube_pos"] = self._cube_xyz()
        info["cube_yaw"] = self._cube_yaw()
        info["target_pos"] = self._cube_goal_xyz
        info["target_yaw"] = self._cube_goal_yaw
        info["success"] = self._success

        phase = self.expert_phases[self._expert_phase_idx]
        info["expert_target"] = self._compute_expert_waypoint(phase).astype(np.float32)
        return info

    def get_step_info(self) -> dict:
        """Returns a dict with the current reset info.
        
         The dict contains:
            goal: the goal image in task_mode
            cube_pos: the current x-y-z position of the cube
            cube_yaw: the current rotation angle of the cube
            target_pos: the target pose of the cube 
            target_yaw: the target orientation of the cube 
            success: current state of the success condition
            expert_target: current or next targeted expert pose 
        """
        info = super().get_step_info()
        if self._goal_image is not None:
            info["goal"] = self._goal_image
        info["cube_pos"] = self._cube_xyz()
        info["cube_yaw"] = self._cube_yaw()
        info["target_pos"] = self._cube_goal_xyz
        info["target_yaw"] = self._cube_goal_yaw
        info["success"] = self._success
        info["expert_target"] = self._advance_expert_phase()
        return info


if __name__ == "__main__":
    env = CloudgripperMuJoCoCube(height=600, width=600)
    try:
        obs, info = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        frame = env.render()
        print(obs, reward, info["cube_pos"], info["target_pos"])
        from PIL import Image
        img = Image.fromarray(frame)
        img.show()

    finally:
        env.close()
