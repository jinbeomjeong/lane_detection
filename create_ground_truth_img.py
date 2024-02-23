import cv2, os
import numpy as np
from datasets import LaneDataset
from tqdm import tqdm

dataset = LaneDataset(data_path="data/val_2.txt")
dataset.I_H = 360
dataset.I_W = 640

output_path = "output_img"


for i, (img, target, path) in enumerate(tqdm(dataset, ncols=50)):

    target = target.numpy().astype(np.uint8)
    target *= 255

    _, file_name = os.path.split(path.replace('\\', '/'))
    file_path = output_path+os.sep+file_name
    cv2.imwrite(file_path, target)
