import os
import time
import cv2
from client.cloudgripper_client import GripperRobot
from cloudgripper_wm.utils.cloudgripper_image_processor import CloudGripperImageProcessor
# --- Config ---
USE_WS = False
ACTION = [1.0, 0.0, 1.0, 0.0, 0.0]

token = os.environ['CLOUDGRIPPER_TOKEN']
robot = GripperRobot('robot23', token)
image_processor = CloudGripperImageProcessor('./cloudgripper_wm/camera_params/cam-to-robot-points/camera-to-robot-cr23.yaml', './cloudgripper_wm/camera_params/base-camera-calibration/camera-params-cr23.yaml')
if USE_WS:
    print("Connecting WebSocket...")
    robot.connect_ws()
    robot._ws.settimeout(None)

cnt = 0
try:
    while True:
        cnt += 1

        t0 = time.time()
        if USE_WS:
            robot.step_action_ws(ACTION)
        else:
            robot.step_action(ACTION)
        print(f"step_action() took {time.time() - t0:.3f} seconds")

        t1 = time.time()
        if USE_WS:
            top_img, _ = robot.get_image_top_ws()
            base_img, _ = robot.get_image_base_ws()
            pass
        else:
            state, _, base_img, _, top_img, _ = robot.get_all_states()

        if top_img is not None:
            cv2.imshow("top", top_img)
        if base_img is not None:
            base_img = image_processor.undistort_fish_eye(base_img)
            cv2.imshow("base", base_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cnt > 100:
            break
finally:
    if USE_WS:
        robot.disconnect_ws()
    cv2.destroyAllWindows()
