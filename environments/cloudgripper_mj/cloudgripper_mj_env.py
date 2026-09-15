from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from dm_control import mjcf
from gymnasium.spaces import Box
from ogbench.manipspace.envs.env import CustomMuJoCoEnv
from ogbench.manipspace.mjcf_utils import safe_find
from dm_control import mjcf

from stable_worldmodel import spaces as swm_spaces





class CloudgripperMuJoCoEnv(CustomMuJoCoEnv):
    """MuJoCo simulation of a CloudGripper cell."""

    metadata: dict = {
        "render_modes": ["rgb_array"],
        "render_fps": 20,
    }

    # Default cloudgripper mujoco model
    folder_path: str = Path(__file__).resolve().parent
    xml_file: str = "cloudgripper_scene.xml"

    # Initial values of default model
    initial_pose: np.ndarray = np.zeros(5, dtype=np.float32)
    material_colors: dict = {
        "glass_floor": np.array([1.0, 1.0, 1.0]),
        "gray_mat": np.array([0.87294, 0.95294, 0.87294]),
        "yellow_plastic": np.array([1.0, 0.8, 0.15]),
        "led_light" : np.array([1.0, 1.0, 1.0]),
        "black_plastic": np.array([0.0, 0.0, 0.0]),
        "seethrough_plastic": np.array([1.0, 1.0, 1.0]),
        "metal": np.array([0.8, 0.8, 0.8]),
    }
    light_intensity: float = 1.5

    # Identifiers of default model
    main_camera: str = "Camera_main"
    camera_names: list = [
        main_camera,
        "Camera_bottom"
    ]
    arm_names: list = [
        "Frame",
        "Rail_slider",
        "Slider",
        "Arm_holder",
        "Arm_linear_gear",
        "Arm_rotation_base",
        "Arm_grip_base",
        "SpurGear1_v1",
        "Arm_grip_finger_right",
        "SpurGear2_v1",
        "Arm_grip_finger_left",
        "Arm_grip_link_back_left",
        "Arm_grip_link_back_right",
        "Arm_grip_link_front_left",
        "Arm_grip_link_front_right",
        "Arm_linear_pinion_gear",
    ]
    servo_names: list = [
        "Servo_gripper",
        "Servo_rotation",
        "Servo_linear",
    ]
    cage_names: list = [
        "Frame",
        "Ground_plate",
    ]
    light_names: list = [
        "wall_right_light_upper",
        "wall_right_light_lower",
        "wall_left_light_upper",
        "wall_left_light_lower",
    ]
    lightbulb_names: list = [
        "led_right_upper1",
        "led_right_upper2",
        "led_right_upper3",
        "led_right_lower1",
        "led_right_lower2",
        "led_right_lower3",
        "led_left_upper1",
        "led_left_upper2",
        "led_left_upper3",
        "led_left_lower1",
        "led_left_lower2",
        "led_left_lower3",
    ]
    wall_names: list = [
        "wall_front",
        "wall_back",
        "wall_right",
        "wall_left",
        "wall_top",
    ]
    material_names: list = [
        "glass_floor",
        "gray_mat",
        "led_light",
        "yellow_plastic",
        "black_plastic",
        "seethrough_plastic",
        "metal",
    ]
    actuator_names: list = [
        "x_actuator",
        "y_actuator",
        "z_actuator",
        "rotate_actuator",
        "gripper_right_actuator",
    ]
    joint_names: list = [
        "Rail_joint",
        "Slider_joint",
        "Linear_joint",
        "Rotation_joint",
        "RightSpur_joint", # remaining gripper joints possess equ. constraints to this one
    ]
    finger_site_names: str = [
        "right_finger_site",
        "left_finger_site", # tool-center-point (tcp) lies between the two
    ]

    def __init__(
        self,
        mode: str = "data_collection",
        max_delta: float = 0.05,
        step_threshold: float = 0.01,
        physics_timestep: float = 0.002,
        control_timestep: float = 0.05,
        **kwargs,
    ):
        """Initialize the wrapper for Cloudgripper MuJoCo model. 

        Args:
            mode: Environment mode (data_collection or task)
            max_delta: Sampling range of action_space for all five joints
            step_threshold: Threshold for moving to a new position for all five joints.
                Mirrors real cloudgripper robot to only actuate joints if the relative
                distance between current and target state exceeds the threshold.
            physics_timestep: Physics timestep.
            control_timestep: Control timestep.
        """
        super().__init__(
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
            **kwargs,
        )

        self._mode: str = mode
        self._max_delta:float = max_delta
        self._step_thresh: float = step_threshold
        self._current_pos:np.ndarray = self.initial_pose.copy()
        self._target_pos: np.ndarray = self.initial_pose.copy()
        

    def build_mjcf_model(self) -> mjcf.RootElement:
        """Loads the default cloudgripper mujoco environment.

        Also extracts names of the model's assets, including
            - body names
            - camera names
            - joint names
            - light names

        mjcf/xml : "environments/mj_cloudgripper/cloudgripper_scene.xml"
        blender  : "environments/mj_cloudgripper/cloudgripper_scene.blend"
        
        World frame is placed in the middle of the ground plate,
        as visualized below. For custom workspace layouts define
        a translation function in swm-child class.

        ______________________
        |                    |
        |                    |
        |                    |
        |             y      |
        |          -->       |
        |         |          |
        |         v          |
        |           x        |
        |                    |
        |________cam_________|
                (main)
        """
        mjcf_model: mjcf.RootElement = mjcf.from_path(str(self.folder_path / self.xml_file))
        
        return mjcf_model

    def verify_mujoco_assets(
            self,
            assets_dict: dict[mujoco.mjtObj, list[str]]
        ) -> dict[mujoco.mjtObj, list[str]]:
        """Checks if specified assets exist in the loaded MjModel.
        
        Args:
            assets_dict: contains names of assets per type
        
        Returns:
            dictionary of assets missing
        """
        missing_assets = {}
        
        for obj_type, names in assets_dict.items():
            missing = []
            for name in names:
                # mj_name2id returns -1 if the asset is not found
                if mujoco.mj_name2id(self.model, obj_type, name) == -1:
                    missing.append(name)
            if missing:
                missing_assets[obj_type.name] = missing
                
        return missing_assets

    def post_compilation(self) -> None:
        """Verifies that all default assets are present in loaded model. 
        
        Use to store information after build of mujoco model.
        """
        expected_assets = {
            mujoco.mjtObj.mjOBJ_BODY: (self.arm_names + self.servo_names + self.cage_names + self.light_names + self.wall_names),
            mujoco.mjtObj.mjOBJ_JOINT: self.joint_names,
            mujoco.mjtObj.mjOBJ_ACTUATOR: self.actuator_names,
            mujoco.mjtObj.mjOBJ_MATERIAL: self.material_names,
            mujoco.mjtObj.mjOBJ_LIGHT: self.lightbulb_names,
            mujoco.mjtObj.mjOBJ_CAMERA: self.camera_names,
        }

        missing_assets = self.verify_mujoco_assets(expected_assets)

        if missing_assets:
            err = [f"{obj_type}: {names}" for obj_type, names in missing_assets.items()]
            raise ValueError(
                 f"Missing assets {err} \n in file {self.folder_path / self.xml_file}"
                )

        # adresses of control joints
        self._active_joint_adrs = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in self.joint_names
        ]

        # fingertip site ids, for get_tcp_world_pos()/solve_tcp_ik()
        self._finger_site_ids = [ 
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )
            for site_name in self.finger_site_names
        ]

        self.post_compilation_objects()
        

    def post_compilation_objects(self) -> None:
        """Use to store information of objects.
        """
        pass

    def compute_observation(self) -> dict:
        """Returns observation of scene.

        Note: camera observations are handled via render function.
        """
        return {
            "state": self._current_pos.astype(np.float32)
        }

    def compute_reward(self) -> float:
        return 0.0

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "state": Box(
                    low=0.0,
                    high=1.0,
                    shape=(5,),
                    dtype=np.float32
                ),
            }
        )

    @property
    def action_space(self):
        """Actions are delta actions by default.
        """
        return swm_spaces.Box(
            low=-self._max_delta,
            high=self._max_delta,
            shape=(5,),
            dtype=np.float32,
        )

    @staticmethod
    def unnormalize(val, range):
        """Unnormalize value depending on joint range
        """
        l_lim, u_lim = range
        return (u_lim - l_lim) * val + l_lim

    def set_active_joints(self, values: np.ndarray) -> None:
        """Controls active joints and solves kinematics for stable configuration.

         Passive joints are linked to active joints via equality constraints, which
         are solved explictly without stepping the environment to enforce forward
         kinematics without computing dynamic effects.

        Active joints are by default
            Rail_joint
            Slider_joint
            Linear_joint
            Rotation_joint
            RightSpur_joint

        Args:
            values: normalized [0, 1] target for each of self.joint_namesW.
        """
        for val, joint_id, adr in zip(values, self.joint_names, self._active_joint_adrs):
            self.data.qpos[adr] = self.unnormalize(val, self.model.joint(joint_id).range)

        # n^2 passes to ensure all passive joints move (e.g. if not ordered in xml) 
        for _ in range(self.model.neq):
            for i in range(self.model.neq):

                # ignore non-equality constraints
                if self.model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                    continue

                adr1 = self.model.jnt_qposadr[self.model.eq_obj1id[i]]
                adr2 = self.model.jnt_qposadr[self.model.eq_obj2id[i]]

                c = self.model.eq_data[i, :5]
                x = self.data.qpos[adr1] - self.model.qpos0[adr1]

                self.data.qpos[adr2] = self.model.qpos0[adr2] + (
                    c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4
                )

        mujoco.mj_forward(self.model, self.data)

    def get_tcp_world_pos(self) -> np.ndarray:
        """World position of the point between the gripper fingers.

        returns:
            position fo the tool-center-point
        """

        right, left = (
            self.data.site_xpos[sid] for sid in self._finger_site_ids
        )

        return (right + left) / 2.0

    def solve_tcp_ik(
        self,
        target_xyz_world: np.ndarray,
        rot_norm: float,
        grip_norm: float,
        x0: np.ndarray | None = None,
        max_iter: int = 10,
        tol: float = 1e-5,
    ) -> np.ndarray:
        """Solve for normalized x-y-z that places the tcp at target_xyz_world,
        holding rotation/gripper fixed at the given values.

         Due to the offset between the tcp and z-rotation axis as well as
         the arc-form gripping motion, the target is calculated in a Gauss-Newton
         loop with a numerically estimated Jacobian instead of a hand-derived IK
         formulation. The loop finds the configuration by simulating forward
         kinematics but restores afterwards --> simulation is not disturbed.
        
        args:
            target_xyz_world: world target x-y-z position
            rot_norm: normalized target rotation angle
            grip_norm: normalized target gripper state
            x0: 
            max_iter: number of iterations to reach tolerance
            tol: target-guess error tolerance
        returns:
            normalized xyz target position
        """
        target = np.asarray(target_xyz_world, dtype=float)
        guess = np.array(
            self._current_pos[:3] if x0 is None else x0, dtype=float
        )

        qpos_backup = self.data.qpos.copy()
        try:
            def fk(xyz_norm: np.ndarray) -> np.ndarray:
                self.set_active_joints([*xyz_norm, rot_norm, grip_norm])
                return self.get_tcp_world_pos().copy()

            pos = fk(guess)
            eps = 1e-3
            for _ in range(max_iter):
                err = target - pos
                if np.linalg.norm(err) < tol:
                    break
                jac = np.zeros((3, 3))
                for i in range(3):
                    probe = guess.copy()
                    probe[i] = np.clip(probe[i] + eps, 0.0, 1.0)
                    step = probe[i] - guess[i]
                    if step == 0.0:
                        # guess[i] is sitting exactly on a joint-range clip
                        # boundary (e.g. 0.0 or 1.0) — the forward probe
                        # can't move, so probe backward instead of leaving
                        # this column zero (which would stop the solver
                        # from ever moving that axis off the boundary).
                        probe[i] = np.clip(guess[i] - eps, 0.0, 1.0)
                        step = probe[i] - guess[i]
                    jac[:, i] = (fk(probe) - pos) / step if step != 0.0 else 0.0
                delta, *_ = np.linalg.lstsq(jac, err, rcond=None)
                guess = np.clip(guess + delta, 0.0, 1.0)
                pos = fk(guess)
        finally:
            self.data.qpos[:] = qpos_backup
            mujoco.mj_forward(self.model, self.data)

        return guess

    def set_tcp_target(
        self,
        target_xyz_world: np.ndarray,
        rot_norm: float | None = None,
        grip_norm: float | None = None,
    ) -> np.ndarray:
        """Move tool-center-point to a Cartesian world position.

         Solve inverse kinematics to set target for 

        args:
            target_xyz_world: target of the tool-center-point in world coordinates
            rot_norm: gripper orientation angle in [0, 1]
            grip_norm: gripper opening angle in [0, 1]
        returns:
            normalized target state of the robot

        Solves the (x, y, z) via solve_tcp_ik() and hands it to the same
        _target_pos that set_control() already drives toward at the real
        robot's constant-velocity, threshold-gated pace — so reaching the
        target still takes the usual step()/set_control() loop, exactly
        like commanding move_xy/move_z on the real robot. rot_norm/grip_norm
        default to their current commanded values if not given.
        """

        rot_norm = float(self._current_pos[3]) if rot_norm is None else rot_norm
        grip_norm = float(self._current_pos[4]) if grip_norm is None else grip_norm

        xyz_norm = self.solve_tcp_ik(
            target_xyz_world, rot_norm, grip_norm, x0=self._current_pos[:3]
        )
        
        self._target_pos = np.array([*xyz_norm, rot_norm, grip_norm], dtype=np.float32)
        return self._target_pos.copy()

    def set_control(self, action: np.ndarray) -> None:
        """Moves the robot given an action if step threshold is exceeded.

         This control principles mirrors the real cloudgripper robot's actuation.
         Note: forward step of mujoco simulation is performed in parent step() 

        Args:
            action: nd.array of action
        """

        # previous target positions are used since they might have stayed below threshold
        self._target_pos = np.clip(self._target_pos + action, 0., 1.)
        dx, dy, dz, dr, dg = np.abs(self._target_pos - self._current_pos)

        next_pos = self._current_pos.copy()

        # mirros real robot actuation by moving if threshold passed (ref. cloudgripper_env.py)
        move_xy = (dx >= self._step_thresh) or (dy >= self._step_thresh)
        next_pos[0] = self._target_pos[0] if move_xy else next_pos[0]
        next_pos[1] = self._target_pos[1] if move_xy else next_pos[1]
        next_pos[2] = self._target_pos[2] if dz >= self._step_thresh else next_pos[2]
        next_pos[3] = self._target_pos[3] if dr >= self._step_thresh else next_pos[3]
        next_pos[4] = self._target_pos[4] if dg >= self._step_thresh else next_pos[4]

        ctrl = np.zeros(len(self.joint_names))
        for val, j_id, a_id in zip(next_pos, self.joint_names, self.actuator_names):

            joint = self._model.joint(j_id)
            actuator = self._model.actuator(a_id)

            next_pos_mj = self.unnormalize(val, joint.range)

            # use act instead of qpos as qpos reaches act through joint kp spring
            # results in lagging & oscillation for joints carrying substantial mass 
            current_pos_mj = self._data.act[actuator.actadr[0]]
            error = next_pos_mj - current_pos_mj

            # distance per control step
            v_max = actuator.ctrlrange[1]
            deadband = v_max * self._control_timestep

            # can't converge closer than one control step --> hold still
            if abs(error) <= deadband:
                ctrl[actuator.id] = 0.0
            else:
                ctrl[actuator.id] = v_max if error > 0.0 else -v_max

        self._data.ctrl[:] = ctrl
        self._current_pos = next_pos


    def render(
        self,
        camera=None,
        *args,
        **kwargs,
    ):
        """Render image of a cloudgripper camera {"Camera_main" or "Camera_bottom"}.
        """
        camera = camera or self.main_camera
        return super().render(camera=camera, *args, **kwargs)

    def close(self):
        """Releases the offscreen renderer's GL context.

        mujoco.Renderer holds a GLFW context that it frees in __del__. If
        that happens during interpreter shutdown instead, module globals
        glfw's cleanup relies on may already be torn down, causing harmless
        but noisy "Exception ignored in: ... TypeError: 'NoneType' object is
        not callable" messages. Calling close() explicitly (e.g. at the end
        of a script) avoids that by freeing the context deterministically
        while the interpreter is still fully alive.
        """
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

