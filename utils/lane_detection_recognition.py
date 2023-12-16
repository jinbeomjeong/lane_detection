left_lane_det_prv = False
left_lane_det_curr = False
left_lane_det_result = False
right_lane_det_prv = False
right_lane_det_curr = False
right_lane_det_result = False


def lane_det_filter(left_lane_det_state=False, right_lane_det_state=False):
    global left_lane_det_prv, left_lane_det_curr, left_lane_det_result, right_lane_det_prv, right_lane_det_curr, \
        right_lane_det_result

    left_lane_det_result = left_lane_det_prv or left_lane_det_state
    left_lane_det_prv = left_lane_det_state

    right_lane_det_result = right_lane_det_prv or right_lane_det_state
    right_lane_det_prv = right_lane_det_state

    return left_lane_det_result, right_lane_det_result
