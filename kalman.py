import numpy as np
import matplotlib.pyplot as plt

# --- defining constants ---
# speed is about 20mm/s this means that per sampling period the robot moves about 2mm
# we take a small percentage of this to be the error/step so we chose 0.2 which is about 10%
# so we decide that sqrt(qp) will be 0.2 so qp is 0.04mm^2
speed_conv_factor = 0.410958904109589 # TO BE CALIBRATED
dist_btwn_wheels = 95 # mm, TO BE CALIBRATED!
Ts = 0.1 # sampling period

# --- parameters to tune ---
qp = 0.25 # variance on position state, need to tune with speed, if thymio is 50 mm/s,
# then error is about 10% so 5 and for 1 Ts that is 0.5mm so 0.5^2 is 0.25mm^2
rp = 25 # variance on position measurement from the camera, std abuot 5mm
# q_nu = 5 # variance on speed state, taken as an estimate from the exercise 8
# r_nu = 5 # variance on speed measurement, taken as estimate from excercise 8
rt = 0.0025 # variance on camera theta measurement, std about 0.05rad
qt = 0.0025 # variance on state theta estimate 0.05rad = about 3 degrees error
# so 0.05^2 is 0.0025


def wrap_angle(theta):
    return (theta+np.pi)%(2*np.pi) - np.pi

# --- odometry calculations ---
def compute_odometry(vl_raw, vr_raw, old_heading):
    # compute linear speeds first
    vl_conv = vl_raw*speed_conv_factor
    vr_conv = vr_raw*speed_conv_factor
    avg_v = (vl_conv + vr_conv) / 2
    # compute angular speed
    omega = (vr_conv - vl_conv) / dist_btwn_wheels
    dtheta = omega*Ts
    new_heading = old_heading+dtheta
    dx = avg_v * np.cos(new_heading)*Ts
    dy = avg_v * np.sin(new_heading)*Ts
    return dx, dy, dtheta, new_heading, avg_v


# --- kalman function ---
def kalman_filter_main(x_est_prev, P_est_prev, vl_raw, vr_raw, robot_pose_cam, detection_valid):
    old_heading = x_est_prev[2,0]
    # odometry is prediction and then adding CV is correction
    dx, dy, dtheta, new_heading, avg_v = compute_odometry(vl_raw, vr_raw, old_heading)
    # x is the direct measurements which in this case are the prediction
    u = np.array([[dx], [dy], [dtheta]])
    # x_est_prev is state estimate of [x,y,theta]
    # define A, the state transition matrix which propagates change in state estimate
    A = np.array([
        [1, 0, -avg_v*Ts*np.sin(new_heading)],
        [0, 1, avg_v*Ts*np.cos(new_heading)],
        [0, 0, 1]
    ])
    Q = np.array([
        [qp, 0, 0],
        [0, qp, 0],
        [0, 0, qt]]) # variance on the state estimates
    x_est_a_priori = x_est_prev + u
    x_est_a_priori[2,0] = wrap_angle(x_est_a_priori[2,0])
    P_est_a_priori = np.dot(A, np.dot(P_est_prev, A.T)) + Q
    if detection_valid: # camera sees the thymio
        y = np.array([[robot_pose_cam[0]], 
                      [robot_pose_cam[1]],
                      [robot_pose_cam[2]]])
        H = np.eye(3) # H defines what part the measurement is actually relevant to (position and velocity or just velocity for example)
        R = np.array([[rp, 0, 0], [0, rp, 0], [0, 0, rt]]) # the covariance matrix of the measurement representing the uncertainty in the measurement value which we define
        i = y - np.dot(H, x_est_a_priori)
        i[2,0] = wrap_angle(i[2,0])
        S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R # H.T is defining which parts of the state estimate the measurement is allowed to correct
        # S is the model's uncertainty projected into the measurement space using H plus the added sensor/measurement noise which we arbitrarily defined
        # thus we try to minimize S as that will give us a larger kalman gain in the next step
        K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S))) # this tells us how much we should correct our prediction based on the new measurement
        x_est = x_est_a_priori + np.dot(K,i) # here K is used to define how much we're going to jump in relation to the measurement difference that we saw
        x_est[2,0] = wrap_angle(x_est[2,0])
        # a big K will cause a big difference as it is placing more importance on i
        P_est = P_est_a_priori - np.dot(K,np.dot(H, P_est_a_priori)) # shrinking the variance in proportion to what we just learned, so a big K shows that the measurement made the model learn a lot
        # we get a big K when the robot learns to rely on just the measurement and not the prediction as it admits that its prediction is based on very noisy odometry   
        return x_est, P_est
    # camera can't see the thymio so we only use odometry
    return x_est_a_priori, P_est_a_priori