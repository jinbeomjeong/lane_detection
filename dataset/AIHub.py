import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

colors = {"white": {"solid": (255, 255, 255), "dotted": (150, 150, 150)},
          "yellow": {"solid": (0, 0, 255), "dotted": (30, 30, 255)},}
          # "blue": {"solid": (255, 0, 0), "dotted": (200, 30, 30)}}

labels = {"white": {"solid": (1, 1, 1), "dotted": (2, 2, 2)},
          "yellow": {"solid": (3, 3, 3), "dotted": (4, 4, 4)},}
          # "blue": {"solid": (5, 5, 5), "dotted": (6, 6, 6)}}

label_dict = {"white": {"solid": 1, "dotted": 2},
              "yellow": {"solid": 3, "dotted": 4},}
              # "blue": {"solid": 5, "dotted": 6}}


class AIHub(Dataset):
    def __init__(self, path, image_set, transforms=None):
        super(AIHub, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        if image_set is 'train':
            self.data_dir = os.path.join(path, "Training")
        elif image_set is 'val':
            self.data_dir = os.path.join(path, "Validation")
        else:
            self.data_dir = os.path.join(path, "Test")
        self.label_dir = os.path.join(self.data_dir, "label") # [라벨] json 이 저장된 디렉토리
        self.image_dir = os.path.join(self.data_dir, "images") # [원천] png, jpg 가 저장된 디렉토리
        self.seg_label_dir = os.path.join(self.data_dir, "seg_label") # segmentation label image 를 저장할 디렉토리, json 데이터를 기반으로 생성
        self.image_set = image_set
        self.transforms = transforms
        self.img_list = []
        self.segLabel_list = []
        self.exist_list = []

        if not os.path.exists(self.seg_label_dir):
            print("Label is going to get generated into dir: {} ...".format(self.seg_label_dir))
            self.generate_label()
        else:
            self.createIndex()

    def createIndex(self):
        '''
        data_list.txt 를 parsing 하여 [원천] image 와 segmentation 정답지, 차선 존재 여부 정보를 파악
        '''
        with open(os.path.join(self.data_dir, "data_list.txt"), "r", newline="\n") as f:
            for data_list in f:
                data = data_list.split(',')
                self.img_list.append(data[0])
                self.segLabel_list.append(data[1])
                exist = []
                for e in data[2:-1]:
                    exist.append(int(e))
                self.exist_list.append(exist)
        print(self.image_set, len(self.img_list))

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_set != 'test':
            segLabel = cv2.imread(self.segLabel_list[idx])[:, :, 0]

            if np.max(segLabel) > 4 and np.min(segLabel) < 0:
                print(self.segLabel_list[idx])
            exist = np.array(self.exist_list[idx])

        else:
            segLabel = None
            exist = None

        sample = {'img': img, # 원본 이미지
                  'segLabel': segLabel, # segmentation 정보가 표시된 이미지, shape: (원본 이미지의 height, 원본 이미지의 width, 1)
                  'exist': exist, # 각 클래스가 이미지에 존재하는지 여부, 예시: [1, 0, 0, 1]
                  'img_name': self.img_list[idx]}

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.img_list)

    def generate_label(self):
        os.makedirs(self.seg_label_dir, exist_ok=True)
        f = open(os.path.join(self.data_dir, "data_list.txt"), "w")

        for label_dir_name in os.listdir(self.label_dir):
            image_dir_path = os.path.join(self.image_dir, label_dir_name)
            label_dir_path = os.path.join(self.label_dir, label_dir_name)

            if not os.path.isdir(image_dir_path):
                print("{} is not existed".format(image_dir_path))
                continue
            if len(os.listdir(image_dir_path)) == 1:
                image_dir_path = os.path.join(image_dir_path, os.listdir(image_dir_path)[0])
            if len(os.listdir(label_dir_path)) == 1:
                label_dir_path = os.path.join(label_dir_path, os.listdir(label_dir_path)[0])

            if not os.path.exists(image_dir_path):
                print("The image directory is not existed: {}".format(image_dir_path))
                continue
            print("Parsing: {}".format(label_dir_path))
            seg_dir_path = os.path.join(self.seg_label_dir, label_dir_name)
            os.makedirs(seg_dir_path, exist_ok=False)

            for json_file_name in os.listdir(label_dir_path):
                if os.path.getsize(os.path.join(label_dir_path, json_file_name)) <= 10:
                    continue
                with open (os.path.join(label_dir_path, json_file_name), "r", encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    img_name = data["image"]["file_name"]
                    annotations = data["annotations"]
                    img_path = os.path.join(image_dir_path, img_name)
                    img = cv2.imread(img_path)
                    seg_img = np.zeros(img.shape)
                    exists = [0] * len(label_dict) * 2

                    for ann in annotations:
                        if ann["class"] != "traffic_lane":
                            continue
                        if ann["attributes"][0]["value"] == "blue":
                            continue
                        label = labels[ann["attributes"][0]["value"]][ann["attributes"][1]["value"]]
                        exists[label_dict[ann["attributes"][0]["value"]][ann["attributes"][1]["value"]] - 1] = 1
                        line_thickness = 5
                        if img.shape[1] == 1920:
                            # image 가로 길이에 따라 차선 segmentation 두께가 달라지도록 조정 (비율에 따른 차선 두께 조정)
                            line_thickness = 8
                        for i in range(1, len(ann["data"])):
                            coord_1 = ann["data"][i - 1]
                            coord_2 = ann["data"][i]
                            # background 영역은 모두 0 로 채워지고 각 클래스의 line segment 는 클래스 번호 (1, 2, 3, 4) 로 채워진 이미지 생성
                            cv2.line(seg_img, (coord_1["x"], coord_1["y"]), (coord_2["x"], coord_2["y"]), color=label,
                                     thickness=line_thickness)

                    seg_label_path = os.path.join(seg_dir_path, img_name[:-3]+"png")
                    if not os.path.exists(seg_label_path):
                        cv2.imwrite(seg_label_path, seg_img)

                    self.img_list.append(img_path)
                    self.segLabel_list.append(seg_label_path)
                    self.exist_list.append(exists)

                    # data_list.txt 에 image/segmentation path 와 차선 존재여부를 기록
                    data_info = img_path + "," + seg_label_path
                    for e in exists:
                        data_info += "," + str(e)
                    data_info += ",\n"
                    f.write(data_info)

        f.close()

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel'] is None:
            segLabel = None
            exist = None
        elif isinstance(batch[0]['segLabel'], torch.Tensor):
            segLabel = torch.stack([b['segLabel'] for b in batch])
            exist = torch.stack([b['exist'] for b in batch])
        else:
            segLabel = [b['segLabel'] for b in batch]
            exist = [b['exist'] for b in batch]

        samples = {'img': img,
                   'segLabel': segLabel,
                   'exist': exist,
                   'img_name': [x['img_name'] for x in batch]}

        return samples
