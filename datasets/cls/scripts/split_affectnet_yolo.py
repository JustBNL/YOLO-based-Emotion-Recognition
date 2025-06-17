import os
import shutil
import random
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 配置
LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CLASS_MAPPING = {"anger": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy", "sad": "sad", "surprise": "surprise", "neutral": "neutral"}
SPLIT_RATIOS = (0.7, 0.2, 0.1)
SRC_DIR = Path("../raw/affectnet")
DST_DIR = Path("../processed/affectnet/images")
LOG_FILE = "error_log.txt"

# 可选参数
RESIZE_TO = (224, 224)  # 设定图像尺寸
CONVERT_GRAYSCALE = False  # 是否转换为灰度图

# 创建目标目录结构
def create_dirs():
    for split in ["train", "val", "test"]:
        for label in LABELS:
            (DST_DIR / split / label).mkdir(parents=True, exist_ok=True)

# 处理图像：灰度+缩放
def process_image(src_dst_pair):
    src_path, dst_path = src_dst_pair
    try:
        img = Image.open(src_path)
        if CONVERT_GRAYSCALE:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img = img.resize(RESIZE_TO)
        img.save(dst_path)
    except Exception as e:
        with open(LOG_FILE, "a") as log:
            log.write(f"Error processing {src_path}: {e}\n")

# 主函数
def split_and_convert():
    create_dirs()
    tasks = []
    with open(LOG_FILE, "w") as log:
        log.write("Error Log\n==========\n")

    for class_name in os.listdir(SRC_DIR):
        if class_name not in CLASS_MAPPING:
            print(f"Skipping unknown label: {class_name}")
            continue

        mapped_name = CLASS_MAPPING[class_name]
        class_path = SRC_DIR / class_name
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        images = []
        for ext in image_extensions:
            images.extend(class_path.glob(ext))

        print(f"{class_name} (as {mapped_name}): found {len(images)} images")
        if len(images) == 0:
            continue

        random.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * SPLIT_RATIOS[0])
        n_val = int(n_total * SPLIT_RATIOS[1])
        n_test = n_total - n_train - n_val

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split, split_images in splits.items():
            for img_path in split_images:
                img_name = img_path.name
                dst_img = DST_DIR / split / mapped_name / img_name
                tasks.append((img_path, dst_img))

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_image, pair) for pair in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            pass

if __name__ == "__main__":
    split_and_convert()
