import os, torch, cv2
import numpy as np
import ujson as json
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class LaneDataset(Dataset):
    I_H = 432
    I_W = 768

    def __init__(self, data_path):
        with open(data_path) as txt:
            lines = txt.read().splitlines()
            
        self.img_list = []

        for line in lines:
            for f in os.listdir(line):
                if '.jpg' in f or '.png' in f:
                    self.img_list.append(line+os.sep+f)

        self.len = len(self.img_list)
        self.data_path = data_path

    def __getitem__(self, index):
        while True:
            img_path = self.img_list[index]

            try:
                img = Image.open(img_path)
                break

            except:
                with open("error_files.txt", 'a') as errlog:
                    errlog.write(img_path+'\n')
                    index = index + 1
            
        w, h = img.size
        label_path = img_path.replace("image", "json").replace('.jpg', '.json').replace('.png', '.json')

        with open(label_path) as json_file:
            json_data = json.load(json_file)

        img_tensor = transforms.functional.to_tensor(transforms.functional.resized_crop(img, h-w//2, 0, w//2, w, (self.I_H,self.I_W)))
        target_map = self.make_gt_map(json_data, w, h)

        return img_tensor, torch.LongTensor(target_map), img_path

    def __len__(self):
        return self.len

    def make_gt_map(self, json_data, original_w, original_h):
        target_map = np.zeros((self.I_H, self.I_W), dtype=np.int32)
        annotation = json_data["annotations"]
        y_offset = original_h - original_w // 2

        for item in annotation:
            obj_class = item["class"]
            if obj_class == "traffic_lane":
                pos = item["data"]
                poly_points = np.array([([pt["x"]*self.I_W/original_w, (pt["y"] - y_offset)*self.I_H/(original_h-y_offset)]) for pt in pos]).astype(np.int32)
                cv2.polylines(target_map, [poly_points], False, 1,10)
            '''
            if obj_class == "stop_line":
                pos = item["data"]
                poly_points = np.array([([pt["x"]*self.I_W/original_w, (pt["y"] - y_offset)*self.I_H/(original_h-y_offset)]) for pt in pos]).astype(np.int32)
                cv2.polylines(target_map, [poly_points], False, 2,10)

            if obj_class == "crosswalk":
                pos = item["data"]
                poly_points = np.array([([pt["x"]*self.I_W/original_w, (pt["y"] - y_offset)*self.I_H/(original_h-y_offset)]) for pt in pos]).astype(np.int32)
                if len(poly_points) == 0:
                    continue
                cv2.fillPoly(target_map, [poly_points], 3)
             '''
        return target_map


if __name__ == "__main__":
    ld = LaneDataset(data_path='data/train.txt')

    for i, (img, target, path) in enumerate(ld):
        print(i)
        plt.imshow(img.permute((1, 2, 0)))
        plt.savefig("a.png")
        plt.imshow(target, vmax=3)
        plt.savefig("b.png")
