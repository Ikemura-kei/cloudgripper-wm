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

# Largest change per step on each axis [x, y, z, rotation / 180, gripper] for the play data (Kei, 2026-11-25): about
# the same physical distance per step on every axis (14 mm in x, y; 6.5 mm of tip height; 45 deg; the gripper opens
# or closes in one step). 0.05 on every axis made a pick-and-place about 155 steps, these make it about 30.
# Pass it as CloudGripperEnv(max_delta=PLAY_MAX_DELTA); do not mix datasets collected with different limits.
PLAY_MAX_DELTA = (0.1, 0.1, 0.25, 0.25, 0.75)
