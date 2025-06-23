#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集拆分功能

affectnet-1
├── anger
├── disgust
├── fear
├── happy
├── neutral
├── sad
└── surprise

拆分为：

affectnet
└── images
    ├── test
    │   ├── angry
    │   ├── disgust
    │   ├── fear
    │   ├── happy
    │   ├── neutral
    │   ├── sad
    │   └── surprise
    ├── train
    │   ├── angry
    │   ├── disgust
    │   ├── fear
    │   ├── happy
    │   ├── neutral
    │   ├── sad
    │   └── surprise
    └── val
        ├── angry
        ├── disgust
        ├── fear
        ├── happy
        ├── neutral
        ├── sad
        └── surprise
"""

import os
import random
from pathlib import Path
from tqdm import tqdm
import shutil

# ───────────────────────────────────────────────────────────────
# 配置
# ───────────────────────────────────────────────────────────────
SRC_DIR = Path("../processed/affectnet-1")  # 处理后的图像数据目录
DST_DIR = Path("../processed/affectnet/images")  # 拆分后的数据集目标目录
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # 数据集拆分比例 (训练集, 验证集, 测试集)
LABEL_MAPPING = {
    "anger": "angry",  # 映射标签
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}
LABELS = list(LABEL_MAPPING.values())  # 获取映射后的标签列表


def create_dirs():
    """
    创建数据集拆分后的目标目录结构
    """
    for split in ("train", "val", "test"):
        for label in LABELS:
            (DST_DIR / split / label).mkdir(parents=True, exist_ok=True)


def split_dataset():
    """
    拆分数据集并按照指定比例将图像移动到目标文件夹
    """
    image_paths = []
    total_files = 0  # 总文件数

    # 遍历源目录，获取所有图片路径，并根据原始标签进行映射
    for raw_label, mapped_label in LABEL_MAPPING.items():
        class_dir = SRC_DIR / raw_label
        if class_dir.is_dir():
            image_files = list(class_dir.glob("*.*"))  # 查找所有图像文件
            print(f"📂 文件夹 {raw_label}: 找到 {len(image_files)} 张图像")
            total_files += len(image_files)  # 累加图像文件数
            for img_path in image_files:
                image_paths.append((img_path, mapped_label))

    # 输出总图像数量
    print(f"\n📊 总共扫描到 {total_files} 张图像\n")

    random.shuffle(image_paths)  # 随机打乱数据集

    n_total = len(image_paths)
    n_train = int(n_total * SPLIT_RATIOS[0])
    n_val = int(n_total * SPLIT_RATIOS[1])

    split_boundaries = (n_train, n_train + n_val)

    tasks = []
    for idx, (img_path, label) in enumerate(image_paths):
        if idx < split_boundaries[0]:
            split = "train"
        elif idx < split_boundaries[1]:
            split = "val"
        else:
            split = "test"
        dst = DST_DIR / split / label / img_path.name
        tasks.append((img_path, dst))

    # 使用进度条处理文件移动
    for img_path, dst in tqdm(tasks, desc="正在移动图像文件", unit="file"):
        dst.parent.mkdir(parents=True, exist_ok=True)  # 创建目标目录
        os.rename(img_path, dst)  # 移动文件到对应目录


def main():
    """
    主程序：打印配置，删除目标文件夹并重新创建，调用拆分和创建目录的函数
    """
    print("🔧 配置：")
    print(f"  数据集拆分比例: {SPLIT_RATIOS}")
    print(f"  标签映射: {LABEL_MAPPING}")
    print(f"  源数据目录: {SRC_DIR}")
    print(f"  目标数据目录: {DST_DIR}\n")

    # 删除目标文件夹并提示
    if DST_DIR.exists():
        print(f"⚠️ 目标目录 {DST_DIR} 已存在，正在删除并重新创建...")
        shutil.rmtree(DST_DIR)

    # 创建目标目录
    DST_DIR.mkdir(parents=True, exist_ok=True)

    create_dirs()  # 创建拆分后的目录结构
    split_dataset()  # 执行数据集拆分
    print("✅ 数据集拆分完成！")


if __name__ == "__main__":
    main()  # 调用主函数
