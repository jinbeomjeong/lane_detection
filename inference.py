import cv2, torch, time
import numpy as np
from module import LaneClassification
from torchvision import transforms
from sklearn.linear_model import RANSACRegressor

device = torch.device('cuda:0')
model = LaneClassification.load_from_checkpoint(checkpoint_path='resnet_50_640_360.ckpt',
                                                map_location='cuda:0')
model.eval()

cap = cv2.VideoCapture("d:\\video\\urban_street.mp4")
ret, frame = cap.read()
h, w = frame.shape[0:2]

fps = 0.0
t0 = time.time()

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

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

cv2.namedWindow(winname='frame', flags=cv2.WINDOW_NORMAL)


@torch.no_grad()
def main():
    global fps, t0, ret, frame, left_lane_est_pos, right_lane_est_pos, lane_center_pos_list, lane_center_pos_arr
    
    while cap.isOpened():
        start.record()
        t0 = time.time()
        ret, frame = cap.read()
        
        if ret:
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
            #print("fps-image_processing: ", fps)

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
            available_left_lane_pos = (extract_left_lane_pos.shape[0]/(input_height-look_ahead_row_pos)) > 0.3

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
                    lane_center_x_pos = ((right_lane_est_pos[i, 1]-left_lane_est_pos[i, 1])/2)+left_lane_est_pos[i, 1]
                    lane_center_pos_list.append([row, lane_center_x_pos])

                lane_center_pos_arr = np.array(lane_center_pos_list, dtype=np.int32)

            for pos in lane_center_pos_arr:
                cv2.circle(img=result_frame, center=(pos[1], pos[0]), radius=1, color=(255, 0, 0), thickness=2)

            cv2.line(img=result_frame, pt1=([vehicle_center_pos, look_ahead_row_pos]), pt2=([vehicle_center_pos, 540]),
                     color=(0, 0, 255), thickness=2)

            left_lane_x_axis_dist = vehicle_center_pos-left_lane_est_pos[left_lane_est_pos.shape[0]-1, 1]
            right_lane_x_axis_dist = right_lane_est_pos[right_lane_est_pos.shape[0]-1, 1]-vehicle_center_pos

            left_lane_departure = left_lane_x_axis_dist < 50
            right_lane_departure = right_lane_x_axis_dist < 50

            cv2.imshow("frame", result_frame)
            end.record()
            torch.cuda.synchronize()
            #print(start.elapsed_time(end))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
