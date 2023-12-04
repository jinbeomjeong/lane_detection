import time
import argparse
import cv2
import torch
import numpy as np
from model import SCNN
# from utils.prob2lines import getLane
from utils.transforms import Resize, Compose, ToTensor, Normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 모델이 gpu 에서 동작하도록 (gpu가 없으면 cpu)
input_size = (512, 288)
net = SCNN(input_size=input_size, pretrained=False)
net.to(device) # 모델이 gpu 에서 동작하도록 (gpu가 없으면 cpu)

mean = (0.3598, 0.3653, 0.3662) # CULane 데이터셋의 mean, std
std = (0.2573, 0.2663, 0.2756)

# mean = (0.485, 0.456, 0.406) # Imagenet 데이터셋의 mean, std
# std = (0.229, 0.224, 0.225)

transform_img = Resize(input_size)
transform_to_net = Compose(ToTensor(), Normalize(mean=mean, std=std))

color = np.array([[255, 255, 255], [150, 150, 150], [0, 0, 255], [30, 30, 255]], dtype='uint8')

mask_img = cv2.imread(filename="mask_image.jpg")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", '-i', type=str, default="demo/demo_video.mp4", help="Path to demo video")
    parser.add_argument("--weight_path", '-w', type=str, default="experiments/AI_Hub_mobilenetv2/mobilenet_v2_test.pth",
                        help="Path to model weights")
    parser.add_argument("--visualize", '-v', default=True, help="Visualize the result")
    parser.add_argument("--exist_threshold", '-e', type=float, default=0.35, help="A confidence threshold")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    video_path = args.video_path
    weight_path = args.weight_path
    exist_threshold = args.exist_threshold

    save_dict = torch.load(weight_path, map_location=device)
    net.load_state_dict(save_dict['net'])
    net.eval()

    cap = cv2.VideoCapture('d:\\video\\img_1580.MOV')

    while cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        # frame = cv2.resize(src=frame, dsize=(960, 540), interpolation=cv2.INTER_AREA)
        print(frame.shape)

        if ret is False or frame is None:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform_img({'img': img})['img']
        x = transform_to_net({'img': img})['img']
        x.unsqueeze_(0)

        seg_pred, exist_pred = net(x.to(device))[:2]
        seg_pred = seg_pred.detach().cpu().numpy()
        seg_pred = seg_pred[0]

        exist_pred = exist_pred.detach().cpu().numpy()

        exist = [1 if exist_pred[0, i] > exist_threshold else 0 for i in range(4)]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lane_img = np.zeros_like(img)

        coord_mask = np.argmax(seg_pred, axis=0)

        for i in range(0, 4):
            if exist_pred[0, i] > exist_threshold:
                lane_img[coord_mask == (i + 1)] = color[0]

        fps = 1 / (time.time() - t0)
        print("fps-image_processing: ", fps)

        lane_img = cv2.resize(lane_img, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        result_frame = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=frame, beta=1., gamma=0.)
        #print(result_frame.shape)
        # result_frame = cv2.addWeighted(src1=mask_img, alpha=0.8, src2=result_frame, beta=1.0, gamma=0.0)

        # for x in getLane.prob2lines_CULane(seg_pred, exist):q
        #     print(x)



        if args.visualize:
            #print([1 if exist_pred[0, i] > exist_threshold else 0 for i in range(4)])
            cv2.imshow("frame", result_frame)
            #cv2.imshow('1', roi(result_frame))


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
