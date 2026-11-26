from kalman import wrap_angle
from local_navigation import prox
import numpy as np

tolerance = 10  # distance tolerance to consider goal reached
gain_angle = 1.0  # gain for angle correction
gain_distance = 0.5  # gain for distance correction
osbtThrH = 20  # high threshold for obstacle detection
obstThrL = 10  # low threshold for obstacle detection


def tracking(pos_x, pos_y, angle, heading_x, heading_y):
    # compute the angle to the goal
    delta_x = heading_x - pos_x
    delta_y = heading_y - pos_y
    drho = np.sqrt(delta_x**2 + delta_y**2)
    dtheta = np.arctan2(delta_y, delta_x)

    angle_diff = wrap_angle((dtheta - angle))
    
    return angle_diff, drho 

def controller(pos_x, pos_y, angle, heading_x, heading_y, v_left, v_right):
    # compute the speed commands based on the angle difference and distance to the goal
    angle_diff, drho = tracking(pos_x, pos_y, angle, heading_x, heading_y)
    left_cmd = v_left - gain_angle * angle_diff + gain_distance * drho
    right_cmd = v_right + gain_angle * angle_diff + gain_distance * drho
    
    return left_cmd, right_cmd

def motion(samples, thymio, v_left, v_right, i, state, capteurs):
    heading_x, heading_y = samples[i]
    
    # Get updated position from vision
    pos_x, pos_y, angle = thymio

    # Calculate distance to current waypoint
    angle_diff, drho = tracking(pos_x, pos_y, angle, heading_x, heading_y)

    # Check whether waypoint has been reached
    if drho < tolerance:

        i += 1

        if i >= len(samples):
            return 0, 0, i  # goal reached


    if state == 0: 
        # switch from goal tracking to obst avoidance if obstacle detected
        if max(capteurs) > obstThrH:
            state = 1
    elif state == 1:
        if max(capteurs) < obstThrL:
            state = 0
            
    if  state == 0 :
        # goal tracking: turn toward the goal
        left_cmd, right_cmd = controller(pos_x, pos_y, angle, heading_x, heading_y, v_left, v_right)
    
    else:
        # obstacle avoidance
        left_cmd, right_cmd = prox(v_left, v_right, capteurs)

    # Saturate motor commands
    left_cmd = np.clip(left_cmd, -300, 300)
    right_cmd = np.clip(right_cmd, -300, 300)

    return left_cmd, right_cmd, i 


