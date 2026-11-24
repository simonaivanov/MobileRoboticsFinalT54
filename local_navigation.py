def prox(v_left, v_right, prox_horizontal):

    w_l = [40,  20, -20, -20, -40]
    w_r = [-40, -20, -20,  20,  40]

    # Scale factors for sensors and constant factor
    sensor_scale = 200
    
    y = [v_left, v_right]
    x = [0,0,0,0,0]
        
    for i in range(len(x)):
        # Get and scale inputs
        x[i] = prox_horizontal[i] // sensor_scale
        
        # Compute outputs of neurons and set motor powers
        y[0] = y[0] + x[i] * w_l[i]
        y[1] = y[1] + x[i] * w_r[i]
    
    # Set motor powers
    left_cmd = y[0]
    right_cmd = y[1]

    return left_cmd, right_cmd