## AI Hub Dataset

### Preparation
1. AI Hub 데이터셋을 다운받습니다.
2. Training 데이터와 Validation 데이터를 구분하여 각 디렉토리 안에 위치시키십니다.
    - Training 과 Validation 의 폴더 이름이 변경되지 않도록 유의합니다.
3. Training/Validation 디렉토리 안에 images 라는 신규 폴더를 만들어 **[원천]** 에 해당되는 zip, tar 파일을 압축 해제합니다.
4. Training/Validation 디렉토리 안에 label 이라는 신규 폴더를 만들어  **[라벨]** 에 해당되는 zip, tar 파일을 압축 해제합니다.
5. `config.py` 의 `Dataset_Path` 에 AI Hub Dataset의 절대 경로를 추가합니다.
6. `dataset/__init__.py` 에 `from .AIHub import AIHub` 를 추가하여 AIHub 데이터셋 클래스가 정상적으로 참조되도록 합니다.
7. experiments/AIHub_Dataset 안에 cfg.json 파일을 이용해 파라미터 조정을 합니다.
8. 아래 training 명령어을 참고하여 모델을 훈련시킵니다.


```
AI_Hub_Dataset
├── Training
│   ├── images
│   │   ├── [원천]c_1280_720_daylight_train_1
│   │   └── ...
│   ├── label
│   │   ├── [라벨]c_1280_720_daylight_train_1
│   │   └── ...
│   ├── ...
├── Validation
│   ├── images
│   │   ├── [원천]c_1280_720_daylight_train_1
│   │   └── ...
│   ├── label
│   │   ├── [라벨]c_1280_720_daylight_train_1
│   │   └── ...
│   ├── ...
```



```
AI_Hub_Dataset
├── Training
   └─── Training

├── Validation
```
<br>

### Training
```shell
python train.py --exp_dir ./experiments/AIHub_Dataset [--resume/-r]
```

#### 데이터셋 가공 방식 
- training 을 시작하면 데이터셋 디렉토리 안에 seg_label 디렉토리가 있는지 확인합니다.
- seg_label 디렉토리가 존재하지 않으면 segmentation 정답지가 없다고 판단하여 데이터를 파싱하기 시작합니다.
- label 폴더 안의 **[라벨]** json 데이터를 파싱하면서 
  - segmentation 정답지를 새로 생성하여 seg_label 디렉토리 안에 저장합니다.
  - exist 의 ground truth 를 생성합니다.
  - **[원천]** image 의 경로와 segmentation image 의 경로, exist 의 ground truth 를 순서대로 data_list.txt 에 기록합니다.
- 데이터 가공이 완료되면 segmentation image 는 seg_label 디렉토리 안에 저장되고, 데이터 정보를 기록한 data_list.txt 가 생성됩니다.
<br><br>
- 이미 seg_label 디렉토리가 존재한다면 data_list.txt 를 파싱하여 추가 생성 없이 바로 훈련을 시작합니다.

<br>

### Testing with a Video
- 테스트 할 비디오를 demo 디렉토리 안에 위치시킵니다.
  - AIHub 데이터셋 이미지의 가로:세로 비율은 16:9, 19:10 이며, 비슷한 비율의 비디오에서 좋은 성능을 보입니다.
- -i 의 인자로 video 의 경로를 지정하여 `video_test.py` 를 실행시킵니다.

```shell
python video_test.py  -i demo/demo_video.mp4 
                      -w experiments/AIHub/AIHub_example.pth 
                      [--visualize / -v]
```
