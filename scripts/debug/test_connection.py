import os
import time
from client.cloudgripper_client import GripperRobot
token = os.environ['CLOUDGRIPPER_TOKEN']
robot = GripperRobot('robot23', token)

import cv2
cnt = 0
while True:
    cnt += 1
    robot.move_xy(0.5, 0.5)  # Move forward at 0.1 m/s
    time.sleep(0.2)
    if cnt == 3:
        robot.move_gripper(0.5)
        time.sleep(0.3)
        robot.rotate(180)
    elif cnt == 5:
        robot.move_gripper(0.4)
        time.sleep(0.3)
        robot.rotate(180)
    elif cnt == 7:
        robot.move_gripper(0.2)
        time.sleep(0.3)
        robot.rotate(180)
    image, timestamp = robot.getImageTop()
    print(image.shape, timestamp)
    cv2.imshow("Cloudgripper top camera stream", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break