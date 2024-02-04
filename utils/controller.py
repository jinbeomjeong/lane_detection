import time

time_current = time.time()
time_prev = time.time()
integral: float = 0.0
error:float = 0.0
error_prev: float = 0.0

def pid_controller(pid_gain=(1.0, 0.01, 0.0), set_point=1.0, measurement=1.0, min_max_output=(-1.0, 1.0)):
    # pid_gain is kp, ki, kd
    global time_current, integral, time_prev, error, error_prev

    time_current = time.time()
    error = set_point - measurement
    p_val = pid_gain[0] * error
    i_val = integral + pid_gain[1] * error * (time_current-time_prev)
    d_val = pid_gain[2] * (error - error_prev) / (time_current-time_prev)
    control_val = p_val + i_val + d_val

    if control_val < min_max_output[0]:
        control_val = min_max_output[0]
    if control_val > min_max_output[1]:
        control_val = min_max_output[1]

    error_prev = error
    time_prev = time_current

    return control_val