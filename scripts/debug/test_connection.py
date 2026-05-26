import os
from client.cloudgripper_client import GripperRobot
token = os.environ['CLOUDGRIPPER_TOKEN']
robot = GripperRobot('robot23', token)

import cv2

while True:
    robot.move_xy(0.5, 0.5)  # Move forward at 0.1 m/s
    image, timestamp = robot.getImageTop()
    print(image.shape, timestamp)
    cv2.imshow("Cloudgripper top camera stream", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break