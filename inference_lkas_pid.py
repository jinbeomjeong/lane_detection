import cv2, torch, time, struct, imagezmq
import numpy as np
from module import LaneClassification
from torchvision import transforms
from sklearn.linear_model import RANSACRegressor
from utils.udp_lib import UDPClient
from utils.controller import pid_controller


udp_handel = UDPClient(address='192.168.137.111', port=6340)
device = torch.device('cuda:0')
model = LaneClassification.load_from_checkpoint(checkpoint_path='resnet_50_640_360.ckpt',
                                                map_location='cuda:0')
model.eval()

image_hub = imagezmq.ImageHub()

fps = 0.0
input_height = 360
input_width = 640

vanishing_row_pos = 220
look_ahead_row_pos = 250
vehicle_center_pos = 315

lane_roi_row_pos = 250

lane_ref_row_pos = np.array(list(range(look_ahead_row_pos, input_height, 5)), dtype=np.int32)

est_left_lane = RANSACRegressor()
est_right_lane = RANSACRegressor()

left_lane_est_pos = np.zeros((2, 2), dtype=np.int32)
right_lane_est_pos = np.zeros((2, 2), dtype=np.int32)
lane_center_pos_list = []
lane_center_pos_arr = np.array([], dtype=np.int32)


cv2.namedWindow(winname='frame', flags=cv2.WINDOW_NORMAL)
font = cv2.FONT_HERSHEY_COMPLEX

elapsed_time = 0.0
start_time = time.time()
frame_idx = 0
t0 = time.time()
lane_slope = 0.0

@ torch.no_grad()
def main():
    global elapsed_time, frame_idx, fps, t0, ret, frame, left_lane_est_pos, right_lane_est_pos, lane_center_pos_list,\
        lane_center_pos_arr, lane_slope

    while True:
        elapsed_time = time.time() - start_time
        host_name, image_byte = image_hub.recv_jpg()
        image_np = np.frombuffer(image_byte, dtype=np.uint8)
        image_hub.send_reply(b'OK')

        frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        frame = cv2.resize(src=frame, dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)
        img_tensor = transforms.functional.to_tensor(frame)
        img_tensor.unsqueeze_(0)
        output = model.forward(img_tensor.to(device))

        lane_img = torch.sigmoid(output['out'])
        lane_img = torch.argmax(lane_img[0], 0)
        lane_img = lane_img.cpu().numpy().astype(np.uint8)

        left_lane_img = lane_img[:, 0:vehicle_center_pos]
        left_lane_pos = np.argwhere(left_lane_img > 0)

        right_lane_img = lane_img[:, vehicle_center_pos:input_width]
        right_lane_pos = np.argwhere(right_lane_img > 0)
        right_lane_pos[:, 1] = right_lane_pos[:, 1] + vehicle_center_pos

        fps = 1 / (time.time() - t0)
        # print("fps-image_processing: ", fps)

        frame_2 = cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_AREA)
        lane_img *= 255
        lane_img = cv2.cvtColor(lane_img, cv2.COLOR_GRAY2RGB)
        result_frame = cv2.addWeighted(src1=lane_img, alpha=0.5, src2=frame_2, beta=1., gamma=0.)

        extract_left_lane_pos = []

        for row in range(look_ahead_row_pos, input_height):
            find_lane = left_lane_pos[:, 0] == row

            if np.any(find_lane):
                left_lane_x_pos = np.max(left_lane_pos[:, 1][find_lane])

                if vehicle_center_pos - left_lane_x_pos < 250:
                    extract_left_lane_pos.append([row, left_lane_x_pos])

        extract_left_lane_pos = np.array(extract_left_lane_pos, dtype=np.int32)
        available_left_lane_pos = (extract_left_lane_pos.shape[0] / (input_height - look_ahead_row_pos)) > 0.3

        if available_left_lane_pos:
            est_left_lane.fit(extract_left_lane_pos[:, 0].reshape(-1, 1),
                              extract_left_lane_pos[:, 1].reshape(-1, 1),
                              sample_weight=5)
            out = est_left_lane.predict(lane_ref_row_pos.reshape(-1, 1))
            out = out.reshape(-1).astype(np.int32)
            left_lane_est_pos = np.vstack([lane_ref_row_pos, out]).T

        for pos in left_lane_est_pos:
            cv2.circle(img=result_frame, center=(pos[1], pos[0]), radius=1, color=(0, 255, 255), thickness=2)

        extract_right_lane_pos = []

        for row in range(look_ahead_row_pos, input_height):
            find_lane = right_lane_pos[:, 0] == row

            if np.any(find_lane):
                right_lane_x_pos = np.min(right_lane_pos[:, 1][find_lane])

                if right_lane_x_pos - vehicle_center_pos < 250:
                    extract_right_lane_pos.append([row, right_lane_x_pos])

        extract_right_lane_pos = np.array(extract_right_lane_pos, dtype=np.int32)
        available_right_lane_pos = (extract_right_lane_pos.shape[0] / (input_height - look_ahead_row_pos)) > 0.3

        if available_right_lane_pos:
            est_right_lane.fit(extract_right_lane_pos[:, 0].reshape(-1, 1),
                               extract_right_lane_pos[:, 1].reshape(-1, 1),
                               sample_weight=5)
            out = est_right_lane.predict(lane_ref_row_pos.reshape(-1, 1))
            out = out.reshape(-1).astype(np.int32)
            right_lane_est_pos = np.vstack([lane_ref_row_pos, out]).T

        for pos in right_lane_est_pos:
            cv2.circle(img=result_frame, center=(pos[1], pos[0]), radius=1, color=(255, 255, 0), thickness=2)

        lane_center_pos_list.clear()

        if available_left_lane_pos and available_right_lane_pos:
            for i, row in enumerate(lane_ref_row_pos):
                lane_center_x_pos = ((right_lane_est_pos[i, 1] - left_lane_est_pos[i, 1]) / 2) + left_lane_est_pos[
                    i, 1]
                lane_center_pos_list.append([row, lane_center_x_pos])

            lane_center_pos_arr = np.array(lane_center_pos_list, dtype=np.int32)

            lane_slope = np.rad2deg(np.arctan2(lane_center_pos_arr[lane_center_pos_arr.shape[0] - 1][0] - lane_center_pos_arr[0, 0],
                                               lane_center_pos_arr[lane_center_pos_arr.shape[0] - 1][1] - lane_center_pos_arr[0, 1]))
            lane_slope -= 90

        for pos in lane_center_pos_arr:
            cv2.circle(img=result_frame, center=(pos[1], pos[0]), radius=1, color=(255, 0, 0), thickness=2)

        cv2.line(img=result_frame, pt1=([vehicle_center_pos, look_ahead_row_pos]), pt2=([vehicle_center_pos, 540]),
                 color=(0, 0, 255), thickness=2)

        left_lane_x_axis_dist = vehicle_center_pos - left_lane_est_pos[left_lane_est_pos.shape[0] - 1, 1]
        right_lane_x_axis_dist = right_lane_est_pos[right_lane_est_pos.shape[0] - 1, 1] - vehicle_center_pos

        lane_rel_pos = right_lane_x_axis_dist - left_lane_x_axis_dist

        steering_control = pid_controller(pid_gain=[1.0, 0.0, 0.0], set_point=0.0, measurement=lane_rel_pos,
                                          min_max_output=[-lane_slope, lane_slope])

        udp_handel.send_msg(struct.pack('d', steering_control))

        fps = 1 / (time.time() - t0)
        t0 = time.time()

        cv2.putText(result_frame, f'Elapsed Time(sec): {elapsed_time: .2f}', (5, 20), font, 0.5, [0, 0, 255], 1)
        cv2.putText(result_frame, f'Process Speed(FPS): {fps: .2f}', (5, 40), font, 0.5, [0, 0, 255], 1)
        cv2.putText(result_frame, f'N of Frame: {frame_idx}', (5, 60), font, 0.5, [0, 0, 255], 1)

        cv2.imshow("frame", result_frame)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
