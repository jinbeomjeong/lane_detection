import cv2, torch, os
import numpy as np
from module import LaneClassification
from torchvision import transforms
from datasets import LaneDataset

output_path = 'predict_img'
device = torch.device('cuda:0')
model = LaneClassification.load_from_checkpoint(checkpoint_path='resnet_50_640_360.ckpt',
                                                map_location='cuda:0')
model.eval()

dataset = LaneDataset(data_path="data/val_2.txt")
dataset.I_H = 360
dataset.I_W = 640

for i, (img_tensor, target, path) in enumerate(dataset):
    img_tensor.unsqueeze_(0)
    output = model.forward(img_tensor.to(device))

    lane_img = torch.sigmoid(output['out'])
    lane_img = torch.argmax(lane_img[0], 0)
    lane_img = lane_img.cpu().numpy().astype(np.uint8)
    lane_img *= 255

    _, file_name = os.path.split(path.replace('\\', '/'))
    file_path = output_path+os.sep+file_name
    cv2.imwrite(file_path, lane_img)

