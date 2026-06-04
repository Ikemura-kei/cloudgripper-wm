import os
import time
import cv2
from client.cloudgripper_client import GripperRobot

# --- Config ---
USE_WS = True
ACTION = [0.4, 0.2, 0.0, 0.0, 0.0]

token = os.environ['CLOUDGRIPPER_TOKEN']
robot = GripperRobot('robot23', token)

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
            # top_img, base_img, state, _ = robot.get_all_states_ws()
            top_img, _ = robot.get_image_top_ws()
            base_img, _ = robot.get_image_base_ws()
            # state, _ = robot.get_state_ws()
            pass
        else:
            state, _, base_img, _, top_img, _ = robot.get_all_states()
        print(f"get_all_states() took {time.time() - t1:.3f} seconds")
        # if state:
        #     print(f"  x={state['x_norm']:.3f}  y={state['y_norm']:.3f}  "
        #           f"z={state['z_norm']:.3f}  rot={state['rotation']:.1f}°  "
        #           f"grip={state['claw_norm']:.3f}")

        if top_img is not None:
            cv2.imshow("top", top_img)
        if base_img is not None:
            cv2.imshow("base", base_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cnt > 100:
            break
finally:
    if USE_WS:
        robot.disconnect_ws()
    cv2.destroyAllWindows()
