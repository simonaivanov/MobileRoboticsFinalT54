import numpy as np
import matplotlib.pyplot as plt

# --- defining constants ---
# speed is about 20mm/s this means that per sampling period the robot moves about 2mm
# we take a small percentage of this to be the error/step so we chose 0.2 which is about 10%
# so we decide that sqrt(qp) will be 0.2 so qp is 0.04mm^2

# since the thymio gradually changes to a stripe of a different shade,
# so the true edge between stripes lies somewhere between the gradual change
# and since there is a displacement of about 0.2mm per sample, we can predict that
# the uncertainty of that estimate is about 0.5mm so sqrt(rp) = 0.5mm so rp=0.25mm^2

# since here rp > qp that means we are trusting the prediction based on the state over one step

# so if filter lags behind the real value we should decrease r so that the thymio trusts the
# measurement more or increase q but if it jumps at every transition then we increase r and decrease q
Ts = 0.1 # sampling period
dist_btwn_wheels = 95 # mm, have to check!!
speed_conv_factor = 0.410958904109589 # taken from the calculation in tp 8

qp = 0.04 # variance on position state
rp = 0.25 # variance on position measurement 
q_nu = 5 # variance on speed state, taken as an estimate from the exercise 8
r_nu = 5 # variance on speed measurement, taken as estimate from excercise 8
r_theta = 0.0025 # variance on camera theta measurement
q_om = 0.05 # variance on angular velocity state
r_om = 0.05 # variance on angular velocity odometry measurement
Q_main = np.array([[qp, 0, 0, 0],[0, qp, 0, 0], [0, 0, q_nu, 0], [0, 0, 0, q_nu]])
Q_theta = np.array([qp,0],[0, q_om])
A_main = np.array([[1,0,Ts,0], [0,1,0,Ts], [0, 0, 1, 0], [0, 0, 0, 1]])
A_theta = np.array([[1,Ts], [0, 1]])

def wrap_angle(theta):
    return (theta+np.pi)%(2*np.pi) - np.pi

# --- odometry calculations ---
def compute_odometry(vl_raw, vr_raw):
    # compute linear speeds first
    vl_conv = vl_raw*speed_conv_factor
    vr_conv = vr_raw*speed_conv_factor
    avg_v = (vl_conv + vr_conv) / 2
    # compute angular speed
    omega = (vr_conv - vl_conv) / dist_btwn_wheels
    heading = wrap_angle(heading+omega*Ts) # need to store heading globally
    vx = avg_v * np.cos(heading)
    vy = avg_v * np.sin(heading)
    return vx, vy, omega, heading


# --- kalman function ---
def kalman_filter_main(x_est_prev, P_est_prev, A_main, Q_main, rp, r_nu, vx, vy, robot_pose_cam, detection_valid):
    x_est_a_priori = np.dot(A_main, x_est_prev)
    P_est_a_priori = np.dot(A_main, np.dot(P_est_prev, A_main.T)) + Q_main
    if detection_valid: # camera sees the thymio
        y = np.array([[robot_pose_cam[0]], 
                      [robot_pose_cam[1]],
                      [vx],
                      [vy]])
        H = np.eye(4) # H defines what part the measurement is actually relevant to (position and velocity or just velocity for example)
        R = np.array([[rp, 0, 0, 0], [0, rp, 0, 0], [0, 0, r_nu, 0], [0, 0, 0, r_nu]]) # the covariance matrix of the measurement representing the uncertainty in the measurement value which we define
    else: # camera doesn't see the thymio so use only the velocities
        y = np.array([[vx], [vy]])
        H = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1]])
        R = np.array([r_nu,0], [0, r_nu])
    i = y - np.dot(H, x_est_a_priori)
    S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R # H.T is defining which parts of the state estimate the measurement is allowed to correct
    # S is the model's uncertainty projected into the measurement space using H plus the added sensor/measurement noise which we arbitrarily defined
    # thus we try to minimize S as that will give us a larger kalman gain in the next step
    K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S))) # this tells us how much we should correct our prediction based on the new measurement
    x_est = x_est_a_priori + np.dot(K,i) # here K is used to define how much we're going to jump in relation to the measurement difference that we saw
    # a big K will cause a big difference as it is placing more importance on i
    P_est = P_est_a_priori - np.dot(K,np.dot(H, P_est_a_priori)) # shrinking the variance in proportion to what we just learned, so a big K shows that the measurement made the model learn a lot
    # we get a big K when the robot learns to rely on just the measurement and not the prediction as it admits that its prediction is based on very noisy odometry
    return x_est, P_est

def kalman_filter_theta(x_est_prev, P_est_prev, omega, theta_cam, detection_valid):
    x_est_a_priori = np.dot(A_theta, x_est_prev)
    P_est_a_priori = np.dot(A_theta, np.dot(P_est_prev, A_theta.T)) + Q_theta
    if detection_valid:
        y = np.array([[theta_cam], [omega]])
        H = np.eye(2)
        R = np.array([[r_theta, 0], [0, r_om]])
    else:
        y = np.array([omega])
        H = np.array([[0,1]])
        R = r_om

    i = y-np.dot(H, x_est_a_priori)
    i[0,0] = wrap_angle(i[0,0])
    S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R # H.T is defining which parts of the state estimate the measurement is allowed to correct
    # S is the model's uncertainty projected into the measurement space using H plus the added sensor/measurement noise which we arbitrarily defined
    # thus we try to minimize S as that will give us a larger kalman gain in the next step
    K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S))) # this tells us how much we should correct our prediction based on the new measurement
    x_est = x_est_a_priori + np.dot(K,i) # here K is used to define how much we're going to jump in relation to the measurement difference that we saw
    # a big K will cause a big difference as it is placing more importance on i
    P_est = P_est_a_priori - np.dot(K,np.dot(H, P_est_a_priori)) # shrinking the variance in proportion to what we just learned, so a big K shows that the measurement made the model learn a lot
    # we get a big K when the robot learns to rely on just the measurement and not the prediction as it admits that its prediction is based on very noisy odometry
    return x_est, P_est
    

# --- kalman implementation that gets called ---
def kalman_step(vl_raw, vr_raw, Ts, robot_pose_cam, detection_valid):
    x_est = [np.array([[0], [0], [0], [0]])] # initial guess is that the thymio starts at (0,0)
    P_est = np.array([[1000, 0, 0, 0],[0, 1000, 0, 0], [0, 0, 1000, 0], [0, 0, 0, 1000]]) # this tells the thymio that the initial estimate is most likely wrong since the covariance is so high and so in the first few runs R is much more important than P so the model will just rely on R
    xt_est = np.array([[0]])
    Pt_est = 1000*np.ones(2)

    # 1. odometry
    vx, vy, omega, heading = compute_odometry(vl_raw, vr_raw)
    # 2. heading filter
    theta_x_est, theta_P_est = kalman_filter_theta(xt_est, Pt_est, omega, robot_pose_cam[2], detection_valid)
    # 3. kalman main filter
    x_est, P_est = kalman_filter_main(x_est, P_est, A_main, Q_main, rp, r_nu, vx, vy, robot_pose_cam, detection_valid)

    return x_est, theta_x_est





