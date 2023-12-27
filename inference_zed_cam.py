import cv2, torch, time, can
import numpy as np
import pandas as pd
from module import LaneClassification
from torchvision import transforms
from sklearn.linear_model import RANSACRegressor
from utils.lane_detection_recognition import lane_det_filter
from utils.system_communication import AdasCANCommunication, ClusterCANCommunication

vehicle_can_ch = can.interface.Bus(bustype='pcan', channel='PCAN_USBBUS1', bitrate=500000)

adas_can_parser = AdasCANCommunication(dbc_filename='resource/ADAS_can_protocol.dbc')
clu_can_parser = ClusterCANCommunication(dbc_filename='resource/Evits_EV_CAN_DBC_CLU_LDWS.dbc')

device = torch.device('cuda:0')
model = LaneClassification.load_from_checkpoint(checkpoint_path='resnet_50_640_360_v2.ckpt',
                                                map_location='cuda:0')
model.eval()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()

fps = 0.0
t0 = time.time()

input_height = 360
input_width = 640

vanishing_row_pos = 220
look_ahead_row_pos = 250
vehicle_center_pos = int(input_width / 2)

lane_ref_row_pos = np.array(list(range(look_ahead_row_pos, input_height, 5)), dtype=np.int32)

est_left_lane = RANSACRegressor()
est_right_lane = RANSACRegressor()

left_lane_est_pos = np.zeros((2, 2), dtype=np.int32)
right_lane_est_pos = np.zeros((2, 2), dtype=np.int32)
lane_center_pos_list = []
lane_center_pos_arr = np.array([], dtype=np.int32)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

cv2.namedWindow(winname='frame', flags=cv2.WINDOW_NORMAL)
font = cv2.FONT_HERSHEY_COMPLEX

logging_data = pd.DataFrame()
logging_header = pd.DataFrame(columns=['time(sec)', 'frame', 'normal_detection', 'left_lane_departure_state',
                                       'right_lane_departure_state', 'period(Hz)'])
start_time_str = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
logging_file_name = 'test' + '_' + start_time_str
logging_file_path = './logging_data/' + logging_file_name + '.csv'
logging_header.to_csv(logging_file_path, mode='a', header=True)

elapsed_time = 0.0
start_time = time.time()
frame_idx = 0
normal_lane_det = 0
left_lane_det = 0
right_lane_det = 0
lane_det_state = 0
msg_list = []


@torch.no_grad()
def main():
    global elapsed_time, fps, t0, ret, frame, left_lane_est_pos, right_lane_est_pos, lane_center_pos_list, \
        lane_center_pos_arr, frame_idx, logging_data, normal_lane_det, left_lane_det, right_lane_det, lane_det_state

    while cap.isOpened():
        elapsed_time = time.time() - start_time
        ret, frame = cap.read()

        if ret:
            left_frame = np.split(frame, 2, axis=1)
            frame = cv2.resize(src=left_frame[0], dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)
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

            frame_2 = cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_AREA)
            lane_img *= 255
            lane_img = cv2.cvtColor(lane_img, cv2.COLOR_GRAY2RGB)
            result_frame = cv2.addWeighted(src1=lane_img, alpha=0.5, src2=frame_2, beta=1., gamma=0.)

            extract_left_lane_pos = []

            for row in range(look_ahead_row_pos, input_height):
                find_lane = left_lane_pos[:, 0] == row

                if np.any(find_lane):
                    left_lane_x_pos = np.max(left_lane_pos[:, 1][find_lane])

                    if vehicle_center_pos - left_lane_x_pos < 200:
                        extract_left_lane_pos.append([row, left_lane_x_pos])

            extract_left_lane_pos = np.array(extract_left_lane_pos, dtype=np.int32)
            available_left_lane_pos = (extract_left_lane_pos.shape[0] / (input_height - look_ahead_row_pos)) > 0.3

            if available_left_lane_pos:
                est_left_lane.fit(extract_left_lane_pos[:, 0].reshape(-1, 1), extract_left_lane_pos[:, 1].reshape(-1, 1),
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

                    if right_lane_x_pos - vehicle_center_pos < 200:
                        extract_right_lane_pos.append([row, right_lane_x_pos])

            extract_right_lane_pos = np.array(extract_right_lane_pos, dtype=np.int32)
            available_right_lane_pos = (extract_right_lane_pos.shape[0] / (input_height - look_ahead_row_pos)) > 0.3

            if available_right_lane_pos:
                est_right_lane.fit(extract_right_lane_pos[:, 0].reshape(-1, 1), extract_right_lane_pos[:, 1].reshape(-1, 1),
                                   sample_weight=5)
                out = est_right_lane.predict(lane_ref_row_pos.reshape(-1, 1))
                out = out.reshape(-1).astype(np.int32)
                right_lane_est_pos = np.vstack([lane_ref_row_pos, out]).T

            for pos in right_lane_est_pos:
                cv2.circle(img=result_frame, center=(pos[1], pos[0]), radius=1, color=(255, 255, 0), thickness=2)

            lane_center_pos_list.clear()

            if available_left_lane_pos and available_right_lane_pos:
                for i, row in enumerate(lane_ref_row_pos):
                    lane_center_x_pos = (((right_lane_est_pos[i, 1] - left_lane_est_pos[i, 1]) / 2)
                                         + left_lane_est_pos[i, 1])
                    lane_center_pos_list.append([row, lane_center_x_pos])

                lane_center_pos_arr = np.array(lane_center_pos_list, dtype=np.int32)

            for pos in lane_center_pos_arr:
                cv2.circle(img=result_frame, center=(pos[1], pos[0]), radius=1, color=(255, 0, 0), thickness=2)

            cv2.line(img=result_frame, pt1=(vehicle_center_pos, look_ahead_row_pos),
                     pt2=(vehicle_center_pos, input_height), color=(0, 0, 255), thickness=2)

            left_lane_x_axis_dist = vehicle_center_pos - left_lane_est_pos[left_lane_est_pos.shape[0] - 1, 1]
            right_lane_x_axis_dist = right_lane_est_pos[right_lane_est_pos.shape[0] - 1, 1] - vehicle_center_pos

            left_lane_departure, right_lane_departure = lane_det_filter(left_lane_x_axis_dist < 130,
                                                                        right_lane_x_axis_dist < 130)
            fps = 1 / (time.time() - t0)
            t0 = time.time()

            if left_lane_departure:
                left_lane_det += 1
                lane_det_state = 1

            if right_lane_departure:
                right_lane_det += 1
                lane_det_state = 2

            if not left_lane_departure and not right_lane_departure:
                normal_lane_det += 1
                lane_det_state = 0

            cv2.putText(result_frame, f'Elapsed Time(sec): {elapsed_time: .2f}', (5, 20), font, 0.5, [0, 0, 255], 1)
            cv2.putText(result_frame, f'Process Speed(FPS): {fps: .2f}', (5, 40), font, 0.5, [0, 0, 255], 1)
            cv2.putText(result_frame, f'N of Frame: {frame_idx}', (5, 60), font, 0.5, [0, 0, 255], 1)
            cv2.putText(result_frame, f'Normal Lane Det.: {normal_lane_det}', (5, 80), font, 0.5, [0, 0, 255], 1)
            cv2.putText(result_frame, f'Left Lane Det.: {left_lane_det}', (5, 100), font, 0.5, [0, 0, 255], 1)
            cv2.putText(result_frame, f'Right Lane Det.: {right_lane_det}', (5, 120), font, 0.5, [0, 0, 255], 1)

            cv2.imshow("frame", result_frame)

            logging_data = pd.DataFrame({'1': round(elapsed_time, 2), '2': frame_idx, '3': normal_lane_det,
                                         '4': left_lane_det, '5': right_lane_det, '6': round(fps, 2)}, index=[0])
            logging_data.to_csv(logging_file_path, mode='a', header=False)
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            msg_list.append(adas_can_parser.create_ldws_can_msg(ldws_state=lane_det_state+1))
            msg_list.append(clu_can_parser.create_ldws_can_msg(ldws_state=lane_det_state))

            for msg in msg_list:
                vehicle_can_ch.send(msg)
                vehicle_can_ch.flush_tx_buffer()

            msg_list.clear()

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
