from kalman import wrap_angle
from global_navigation import plan_path
from local_navigation import prox
from vision import detect_thymio
import numpy as np


def tracking(pos_x, pos_y, angle, heading_x, heading_y):
    # compute the angle to the goal
    delta_x = heading_x - pos_x
    delta_y = heading_y - pos_y
    drho = np.sqrt(delta_x**2 + delta_y**2)
    dtheta = np.arctan2(delta_y, delta_x)

    angle_diff = wrap_angle((dtheta - angle))
    
    return angle_diff, drho 

def controller(pos_x, pos_y, angle, heading_x, heading_y, Gain_angle, Gainp, v_left, v_right):
    # compute the speed commands based on the angle difference and distance to the goal
    angle_diff, drho = tracking(pos_x, pos_y, angle, heading_x, heading_y)
    left_cmd = v_left - Gain_angle * angle_diff + Gainp * drho
    right_cmd = v_right + Gain_angle * angle_diff + Gainp * drho
    
    return left_cmd, right_cmd

def motion(obstacles, thymio_pos, goal, arena_w, arena_h, frame, H, v_left, v_right, 
           obstThrH, obstThrL, Gain_angle, Gainp, tolerance, i, state, capteurs):
    result = plan_path(obstacles, thymio_pos, goal, arena_w, arena_h, algo="A*")
    path = result["waypoints"]
    heading_x, heading_y = path[i]
    
    # Get updated position from vision
    pos_x, pos_y, angle = detect_thymio(frame, H)

    # Calculate distance to current waypoint
    angle_diff, drho = tracking(pos_x, pos_y, angle, heading_x, heading_y)

    # Check whether waypoint has been reached
    if drho < tolerance:

        i += 1

        if i >= len(path):
            return i  # goal reached


    if state == 0: 
        # switch from goal tracking to obst avoidance if obstacle detected
        if max(capteurs) > obstThrH:
            state = 1
    elif state == 1:
        if max(capteurs) < obstThrL:
            state = 0
            
    if  state == 0 :
        # goal tracking: turn toward the goal
        left_cmd, right_cmd = controller(pos_x, pos_y, angle, heading_x, heading_y, Gain_angle, Gainp, v_left, v_right)
    
    else:
        # obstacle avoidance
        left_cmd, right_cmd = prox(v_left, v_right, capteurs)

    # Saturate motor commands
    left_cmd = np.clip(left_cmd, -300, 300)
    right_cmd = np.clip(right_cmd, -300, 300)

    return left_cmd, right_cmd, i 

