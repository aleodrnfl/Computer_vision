import os
import json
from glob import glob
from tqdm import tqdm
import shutil

# 사용자 설정
LABELME_IMG_DIR = "/CV_Train_Aug/Images"
LABELME_LABEL_DIR = "/CV_Train_Aug/Labels"
YOLO_ROOT = "yolo_dataset"

# YOLO 폴더 구조 생성
os.makedirs(f"{YOLO_ROOT}/images/train", exist_ok=True)
os.makedirs(f"{YOLO_ROOT}/labels/train", exist_ok=True)

# 클래스 정의
CLASS_LIST = ["standing", "lying", "throwing", "sitting"]
LABEL_TO_ID = {k: i for i, k in enumerate(CLASS_LIST)}

# 파일 리스트
img_files = sorted(glob(os.path.join(LABELME_IMG_DIR, "*.png")))
json_files = [os.path.join(LABELME_LABEL_DIR, os.path.splitext(os.path.basename(f))[0] + ".json") for f in img_files]

print(f"전체 훈련 데이터: {len(img_files)}개")

# 모든 데이터 처리
for img_path, json_path in tqdm(zip(img_files, json_files), desc="데이터 처리 중"):
    img_out = f"{YOLO_ROOT}/images/train/{os.path.basename(img_path)}"
    label_out = f"{YOLO_ROOT}/labels/train/{os.path.splitext(os.path.basename(img_path))[0]}.txt"

    # 이미지 복사
    shutil.copy2(img_path, img_out)

    # JSON → YOLO txt 변환
    with open(json_path, "r") as f:
        data = json.load(f)
    img_w, img_h = data["imageWidth"], data["imageHeight"]
    lines = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in LABEL_TO_ID:
            continue  # 예외 처리
        class_id = LABEL_TO_ID[label]
        points = shape["points"]
        x1, y1 = points[0]
        x2, y2 = points[1]
        # 정렬 보정
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        # 중심, 크기 계산
        x_c = (x_min + x_max) / 2 / img_w
        y_c = (y_min + y_max) / 2 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    # YOLO txt 저장
    with open(label_out, "w") as f:
        f.write("\n".join(lines))

# data.yaml 생성
yaml_path = os.path.join(YOLO_ROOT, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"train: {YOLO_ROOT}/images/train\n")
    f.write(f"val: {YOLO_ROOT}/images/train\n")  # 필요시 val 경로로 변경
    f.write("nc: 4\n")
    f.write("names: ['standing', 'lying', 'throwing', 'sitting']\n")

print("YOLO 데이터셋 변환 및 폴더 구조 생성 완료!")
print(f"data.yaml 경로: {yaml_path}")