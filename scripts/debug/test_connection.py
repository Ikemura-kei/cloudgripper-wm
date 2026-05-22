import os
from client.cloudgripper_client import GripperRobot
token = os.environ['CLOUDGRIPPER_TOKEN']
robot = GripperRobot('robot23', token)

import cv2

while True:
    image, timestamp = robot.getImageTop()
    print(image.shape, timestamp)
    cv2.imshow("Cloudgripper top camera stream", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break