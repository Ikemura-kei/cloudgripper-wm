"""Shared constants for clipping the CloudGripper target position.

Used by both CloudGripperEnv and SafeCloudGripperWrapper so the two stay
in sync.
"""

X_RANGE = (0.175, 0.825)
Y_RANGE = (0.175, 0.825)
GRIPPER_RANGE = (0.0, 1.0)

# World-frame (x, y) bounds of the reachable workspace, in the same
# normalized [0,1]-ish units as `_target_pos` / occupancy heightmaps. The
# physical wall sits just outside this square. Used by
# SafeCloudGripperWrapper to detect objects pushed against the wall.
WORKSPACE_BOUNDS = (0.0, 1.0)
