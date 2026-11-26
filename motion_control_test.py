from math import atan2, cos, hypot

from kalman import wrap_angle


LOOKAHEAD_RADIUS = 55.0  # mm
KP_DISTANCE = 2.0        # motor units / mm
KP_ANGLE = 100.0         # motor units / rad
ARRIVAL_TOLERANCE = 10.0  # mm
MOTOR_LIMIT = 300        # motor units


def follow_trajectory(samples, robot_pose, progress_index=0):
    """Return (left, right, progress_index, target, arrived) for this frame."""
    if robot_pose is None or samples is None or len(samples) == 0:
        return 0, 0, progress_index, None, False

    x, y, theta = robot_pose
    last_index = len(samples) - 1
    while True:
        target = samples[progress_index]
        dx, dy = target[0] - x, target[1] - y
        distance = hypot(dx, dy)
        if distance > LOOKAHEAD_RADIUS or progress_index == last_index:
            break
        progress_index += 1

    if progress_index == last_index and distance <= ARRIVAL_TOLERANCE:
        return 0, 0, progress_index, target, True

    angle_error = wrap_angle(atan2(dy, dx) - theta)
    forward = KP_DISTANCE * distance * max(0.0, cos(angle_error))
    turn = KP_ANGLE * angle_error
    # In the camera frame, positive angles turn clockwise (Y points down).
    left = round(max(-MOTOR_LIMIT, min(MOTOR_LIMIT, forward + turn)))
    right = round(max(-MOTOR_LIMIT, min(MOTOR_LIMIT, forward - turn)))
    return left, right, progress_index, target, False
