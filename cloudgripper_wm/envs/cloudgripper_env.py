import os
import threading
from contextlib import nullcontext
import time
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from cloudgripper_wm.envs.robot_pool import RobotPool
from cloudgripper_wm.tasks import get_task
from cloudgripper_wm.tasks.base import DEFAULT_HOME_POS, Task


class CloudGripperEnv(gym.Env):
    """Single-robot CloudGripper Gymnasium environment.

    Acquires a robot name from RobotPool on init and releases it on close().
    Configure the pool before creating any envs:

        RobotPool.configure(["robot23"])          # 1 robot
        RobotPool.configure(["robot1", ..., "robot8"])  # 8 robots

    Images are exposed via render() (top camera → "pixels" via MegaWrapper).
    The observation dict only carries "state" so MegaWrapper's
    EverythingToInfoWrapper never collides with the "pixels" key AddPixelsWrapper
    already wrote into info.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(
        self,
        token: str | None = None,
        task: str | None = None,
        max_delta: float = 0.05,
        dwell_time: float = 0.5,
        reset_dwell_time: float = 2.0,
        render_mode: str | None = None,
        use_mock: bool = False,
        show_display: bool = False,
        dof_wise: bool = True,
        use_cur_state: bool = False,
        use_ws: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_delta = max_delta
        self.dwell_time = dwell_time
        self.reset_dwell_time = reset_dwell_time
        self.task: Task | None = get_task(task)

        self._robot_name = RobotPool.acquire()

        if use_mock:
            from client.cloudgripper_client_mock import GripperRobotMock
            self.robot = GripperRobotMock(self._robot_name, token or "mock")
        else:
            from client.cloudgripper_client import GripperRobot
            tok = token or os.environ["CLOUDGRIPPER_TOKEN"]
            self.robot = GripperRobot(self._robot_name, tok)

        self.action_space = spaces.Box(
            low=-max_delta,
            high=max_delta,
            shape=(5,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32),
        })

        self.dof_wise = dof_wise
        self.use_cur_state = use_cur_state
        self.use_ws = use_ws
        if use_ws:
            self.robot.connect_ws()
            self.robot._ws.settimeout(None)
        self._target_pos: np.ndarray = DEFAULT_HOME_POS.copy()
        self._last_sent_pos: np.ndarray = np.full(5, np.inf, dtype=np.float32)
        self._last_top_img: np.ndarray | None = None
        self._display_thread: threading.Thread | None = None
        self._display_stop = threading.Event()
        if show_display:
            self._img_lock = threading.Lock()
            self._display_thread = threading.Thread(
                target=self._display_loop, daemon=True, name=f"display-{self._robot_name}"
            )
            self._display_thread.start()
        else:
            self._img_lock = nullcontext()

    # ------------------------------------------------------------------
    # Core Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        self._target_pos = (
            self.task.home_pos().copy() if self.task is not None else DEFAULT_HOME_POS.copy()
        )
        self._last_sent_pos[:] = np.inf   # force all axes to be sent on reset
        print(f"[{self._robot_name}] reset → home={self._target_pos}, dwell={self.reset_dwell_time}s")
        self._send_absolute()

        time.sleep(self.reset_dwell_time)

        top_img, base_img, state = self._observe()
        with self._img_lock:
            self._last_top_img = top_img
        info = {"pixels_base": base_img}
        if self.use_cur_state:
            info["target_state"] = self._target_pos.copy()
        obs = {"state": state}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        start_time = time.time()
        
        action = np.asarray(action, dtype=np.float32)
        self._target_pos = np.clip(self._target_pos + action, 0.0, 1.0)
        self._target_pos[4] = np.clip(self._target_pos[4], 0.2, 0.825)  # gripper range [0.2, 0.8]
        self._send_absolute()

        print("Sending action took {:.3f} seconds, now dwelling for {:.3f} seconds...".format(
            time.time() - start_time, self.dwell_time
        ))

        time.sleep(self.dwell_time)

        obs_start_time = time.time()
        top_img, base_img, state = self._observe()
        with self._img_lock:
            self._last_top_img = top_img
        print(f"observe() took {time.time() - obs_start_time:.3f} seconds")

        obs = {"state": state}
        info: dict = {"pixels_base": base_img}
        if self.use_cur_state:
            info["target_state"] = self._target_pos.copy()
        
        if self.task is not None:
            reward = self.task.compute_reward(obs, action, info)
            terminated = self.task.check_terminated(obs, info)
            info["success"] = self.task.check_success(obs, info)
            info.update(self.task.get_task_info(obs))
        else:
            reward = 0.0
            terminated = False

        print(f"Step completed in {time.time() - start_time:.3f} seconds")

        return obs, reward, terminated, False, info

    def render(self) -> np.ndarray:
        with self._img_lock:
            img = self._last_top_img
        if img is None:
            img, _ = self.robot.getImageTop()
            with self._img_lock:
                self._last_top_img = img
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._display_thread is not None:
            self._display_stop.set()
            self._display_thread.join(timeout=2.0)
        if self.use_ws:
            self.robot.disconnect_ws()
        RobotPool.release(self._robot_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _display_loop(self) -> None:
        """Background thread: show the latest top camera image via cv2.imshow."""
        while not self._display_stop.is_set():
            with self._img_lock:
                img = self._last_top_img
            if img is not None:
                cv2.imshow(self._robot_name, img)
            cv2.waitKey(10)  # 30 ms → ~33 fps max; also pumps the OpenCV event loop
        cv2.destroyWindow(self._robot_name)

    def _send_absolute(self) -> None:
        start_time = time.time()
        x, y, z, rot_norm, grip = self._target_pos
        d = np.abs(self._target_pos - self._last_sent_pos)

        if self.dof_wise:
            move_xy      = self.robot.move_xy_ws      if self.use_ws else self.robot.move_xy
            move_z       = self.robot.move_z_ws       if self.use_ws else self.robot.move_z
            rotate       = self.robot.rotate_ws       if self.use_ws else self.robot.rotate
            move_gripper = self.robot.move_gripper_ws if self.use_ws else self.robot.move_gripper
            if d[0] >= 0.01 or d[1] >= 0.01:
                move_xy(float(x), float(y))
                self._last_sent_pos[0] = x
                self._last_sent_pos[1] = y
                time.sleep(0.01)
            if d[2] >= 0.01:
                move_z(float(z))
                self._last_sent_pos[2] = z
                time.sleep(0.01)
            if d[3] >= 0.01:
                rotate(int(float(rot_norm) * 180))
                self._last_sent_pos[3] = rot_norm
                time.sleep(0.01)
            if d[4] >= 0.01:
                move_gripper(float(grip))
                self._last_sent_pos[4] = grip
                time.sleep(0.01)
        else:
            step_action = self.robot.step_action_ws if self.use_ws else self.robot.step_action
            step_action([float(x), float(y), float(z), float(rot_norm) * 180, float(grip)])
            self._last_sent_pos[:] = self._target_pos

        # print(f"Action sent ({'dof_wise' if self.dof_wise else 'step_action'}) in {time.time() - start_time:.3f} seconds")

    def _observe(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fetch images and state. Returns (top_img_bgr, base_img_rgb, state_array)."""
        if self.use_ws:
            if self.use_cur_state:
                top_img, base_img_raw, state_dict, _ = self.robot.get_all_states_ws()
                state = self._state_dict_to_array(state_dict)
            else:
                top_img, _ = self.robot.get_image_top_ws()
                base_img_raw, _ = self.robot.get_image_base_ws()
                state = self._target_pos.copy()
        else:
            if self.use_cur_state:
                state_dict, _, base_img_raw, _, top_img, _ = self.robot.get_all_states()
                state = self._state_dict_to_array(state_dict)
            else:
                top_img, _ = self.robot.getImageTop()
                base_img_raw, _ = self.robot.getImageBase()
                state = self._target_pos.copy()
        base_img = cv2.cvtColor(base_img_raw, cv2.COLOR_BGR2RGB)
        return top_img, base_img, state

    def _state_dict_to_array(self, state_dict: dict) -> np.ndarray:
        """Convert a state dict from get_all_states() to [x, y, z, rot_norm, gripper] in [0,1]."""
        assert state_dict is not None, f"get_all_states() returned None state for {self._robot_name}"
        return np.array([
            state_dict['x_norm'],
            state_dict['y_norm'],
            state_dict['z_norm'],
            state_dict['rotation'] / 180.0,
            state_dict['claw_norm'],
        ], dtype=np.float32)
