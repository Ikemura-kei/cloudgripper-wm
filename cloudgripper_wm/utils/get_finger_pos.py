import numpy as np


def get_finger_pos(x_c_rob, y_c_rob, z_c_rob, theta, w_grip):
    ROB_REAL_RATIO = 1.0 / 0.14
    W_MAX_REAL = 0.032
    W_MIN_REAL = 0.009

    z_left_rob = z_c_rob
    z_right_rob = z_c_rob

    w_real = w_grip * (W_MAX_REAL - W_MIN_REAL) + W_MIN_REAL
    w_rob = w_real * ROB_REAL_RATIO
    r_rob = w_rob / 2.0

    x_right_rob = x_c_rob + r_rob * np.cos(theta)
    y_right_rob = y_c_rob + r_rob * np.sin(theta)
    x_left_rob = x_c_rob - r_rob * np.cos(theta)
    y_left_rob = y_c_rob - r_rob * np.sin(theta)

    return (x_left_rob, y_left_rob, z_left_rob), (x_right_rob, y_right_rob, z_right_rob)

if __name__ == "__main__":
    x_c_rob = 0.5
    y_c_rob = 0.5
    z_c_rob = 0.5
    w_grip = 1.0
    theta = np.pi / 2.0

    left_pos, right_pos = get_finger_pos(x_c_rob, y_c_rob, z_c_rob, theta, w_grip)
    print("Center position:", (x_c_rob, y_c_rob, z_c_rob))
    print("Left finger position:", left_pos)
    print("Right finger position:", right_pos)