## AI Hub Dataset

### Preparation
1. AI Hub 데이터셋을 다운받습니다.
2. Train 데이터와 Validation 데이터를 구분하여 각 디렉토리 안에 위치시키십니다.
    - `Training` 과 `Validation` 의 폴더 이름이 변경되지 않도록 유의합니다.
3. `Training`/`Validation` 디렉토리 안에 각각 `images` 와 `label` 이라는 신규 폴더를 만듭니다.
4. `images` 디렉토리 안에 **[원천]** 에 해당되는 zip, tar 파일을 압축 해제합니다.
   - zip, tar 파일 선택 후 마우스 우클릭 -> Extract to -> `images` 폴더 선택 
   - **(필수)** 압축해제 후, 폴더명에서 **"[원천]"** 을 부분 삭제합니다.
   - `Training`/`Validation` 디렉토리 안에서 각각 실행
5. `label` 디렉토리 안에  **[라벨]** 에 해당되는 zip, tar 파일을 압축 해제합니다.
    - zip, tar 파일 선택 후 마우스 우클릭 -> Extract to -> `label` 폴더 선택 
    - **(필수)** 압축해제 후, 폴더명에서 **"[라벨]"** 을 부분 삭제합니다.
    - `Training`/`Validation` 디렉토리 안에서 각각 실행
6. `config.py` 의 `Dataset_Path` 에 AI Hub Dataset의 절대 경로를 추가합니다.
    ```
   Dataset_Path = dict(
    CULane="/home/user_name/Dataset/CULane",
    Tusimple="/home/user_name/Dataset/tusimple",
    AIHub=""  # set your AI Hub Dataset Path, "/home/user_name/Dataset/AI_Hub_Dataset")
   ```
7. `dataset/__init__.py` 에 `from .AIHub import AIHub` 를 추가하여 AIHub 데이터셋 클래스가 정상적으로 참조되도록 합니다.
8. experiments/AI_Hub_Dataset 안에 cfg.json 파일을 이용해 파라미터 조정을 합니다.
9. 아래 training 명령어을 참고하여 모델을 훈련시킵니다.


### AI Hub Dataset 디렉토리 구조 예시 (참고)
preparation 을 참고하여 아래 디렉토리 구조를 따르도록 변경
```
AI_Hub_Dataset
├── Training
│   ├── images
│   │   ├── c_1280_720_daylight_train_1
│   │   └── ...
│   ├── label
│   │   ├── c_1280_720_daylight_train_1
│   │   └── ...
│   ├── ...
├── Validation
│   ├── images
│   │   ├── c_1280_720_daylight_validation_1
│   │   └── ...
│   ├── label
│   │   ├── c_1280_720_daylight_validation_1
│   │   └── ...
│   ├── ...
```
<br>

### Training
```shell
python train.py --exp_dir ./experiments/AI_Hub_Dataset [--resume/-r]
```

### 데이터셋 가공 방식 설명 (참고)
- training 을 시작하면 데이터셋 디렉토리 안에 seg_label 디렉토리와 data_list.txt 가 있는지 확인합니다.
- seg_label 디렉토리가 존재하지 않으면 segmentation 정답지가 없다고 판단하여 데이터를 파싱하기 시작합니다.
- label 폴더 안의 json 데이터를 파싱하면서 
  - segmentation 정답지 이미지를 새로 생성하여 seg_label 디렉토리 안에 저장합니다.
    - 각 class 의 line segment를 class 번호(1, 2, 3, 4)로 채운 이미지
  - exist 의 ground truth 를 생성합니다.
    - exist 예시: [0, 1, 0, 1] / 해당 class 가 현재 이미지에서 존재하는지 여부
  - 원본 image의 경로와 segmentation image(정답지)의 경로, exist 정보를 순서대로 data_list.txt 에 기록
- 데이터 가공이 완료되면 segmentation image 는 seg_label 디렉토리 안에 저장되고, 데이터 정보를 기록한 data_list.txt 가 생성됩니다.
<br><br>
- 이미 seg_label 디렉토리와 data_list.txt 가 존재한다면 데이터 추가 생성 없이 바로 훈련을 시작합니다.

### Backbone 모델 변경
`models.py` 안의 net_init 함수의 코드를 변경하여 backbone 모델을 변경합니다. <br>
원하는 모델을 선택하여 주석 해제합니다. (mobilenet_v3 계열, mobilenet_v2 계열, vgg 계열) <br>
torchvision 의 version 에 따라 사용 가능한 모델 리스트가 달라지며 자세한 내용은 [[torchvision models]](https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights)를 참고해 주세요.
```python
def net_init(self, input_size, ms_ks):
    input_w, input_h = input_size
    self.fc_input_feature = 5 * int(input_w/16) * int(input_h/16)
    # 원하는 모델을 선택하여 주석 해제
    # self.backbone = models.vgg16_bn(pretrained=self.pretrained).features # 논문에서 사용된 default 모델
    # self.backbone = models.mobilenet_v3_small(pretrained=self.pretrained).features
    # self.backbone = models.mobilenet_v2(pretrained=self.pretrained).features
```

Gtx 1080 Ti 에서 inference 하였을 때,
1. mobilenet_v3_small backbone: 30fps
2. mobilenet_v2 backbone: 30fps 
3. vgg 16 backbone: 20fps

<br>

### Testing with a Video
- 테스트 할 비디오를 demo 디렉토리 안에 위치시킵니다.
  - AIHub 데이터셋 이미지의 가로:세로 비율은 16:9, 19:10 이며, 비슷한 비율의 비디오에서 좋은 성능을 보입니다.
- -i 의 인자로 video 의 경로를 지정하여 `video_test.py` 를 실행시킵니다.

```shell
python video_test.py  -i demo/demo_video.mp4 
                      -w experiments/AI_Hub_Dataset/AIHub_example.pth 
                      [--visualize / -v]
```
