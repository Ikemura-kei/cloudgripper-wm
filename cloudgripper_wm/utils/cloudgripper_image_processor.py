import yaml
import numpy as np
import cv2
from .coordinate_converter import CoordinateConverter

class CloudGripperImageProcessor:
    def __init__(self, robot_data, cam_params_yaml):
        self.converter = CoordinateConverter(robot_data)
        
        # Load camera parameters for undistortion
        with open(cam_params_yaml, 'r') as file:
            data = yaml.safe_load(file)
        self.mtx_camdown = np.array(data['K'])
        self.dist_camdown = np.array(data['D'])

    def undistort_fish_eye(self, img):
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.mtx_camdown, self.dist_camdown, np.eye(3), self.mtx_camdown, img.shape[:2][::-1], cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img

    def px_py_to_x_y(self, px, py):
        return self.converter.px_py_to_x_y(px, py)

    def x_y_to_px_py(self, x, y):
        return self.converter.x_y_to_px_py(x, y)
