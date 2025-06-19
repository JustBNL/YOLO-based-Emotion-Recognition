#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像放缩与灰度转换功能
"""

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil

# ───────────────────────────────────────────────────────────────
# 配置
# ───────────────────────────────────────────────────────────────
RESIZE_TO = (224, 224)  # 图像目标尺寸
CONVERT_GRAYSCALE = True  # 是否将图像转换为灰度图
SRC_DIR = Path("../raw/affectnet")  # 原始数据目录
DST_DIR = Path("../processed/affectnet-1/images")  # 处理后的图像目录
LOG_FILE = "error_log.txt"  # 错误日志文件路径
IGNORE_FOLDERS = ["contempt"]  # 忽略的文件夹列表 (可以根据需要修改)
INCLUDE_FOLDERS = []  # 仅包含的文件夹列表 (空则表示处理所有文件夹)


# ───────────────────────────────────────────────────────────────
# 辅助函数
# ───────────────────────────────────────────────────────────────

def process_image(src: Path, dst: Path) -> None:
    """
    处理图像（调整大小和转换灰度）
    """
    try:
        img = Image.open(src)
        if CONVERT_GRAYSCALE:
            img = img.convert("L")  # 转为灰度图
        else:
            img = img.convert("RGB")  # 保持为RGB彩色图

        img = img.resize(RESIZE_TO)  # 调整大小
        img.save(dst)  # 保存处理后的图像
    except Exception as exc:
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"❌ 处理失败 {src}: {exc}\n")


def process_images():
    """
    处理所有图像
    """
    total_files = 0  # 用于统计总文件数

    # 打印配置及文件夹结构信息
    print(f"🔧 配置：")
    print(f"  目标尺寸: {RESIZE_TO}")
    print(f"  转换灰度图: {CONVERT_GRAYSCALE}")
    print(f"  源数据目录: {SRC_DIR}")
    print(f"  目标数据目录: {DST_DIR}")
    print(f"  忽略的文件夹: {IGNORE_FOLDERS}")
    print(f"  仅包含的文件夹: {INCLUDE_FOLDERS if INCLUDE_FOLDERS else '所有文件夹'}\n")

    print(f"🔍 正在扫描源数据目录的文件夹结构...")

    # 如果目标文件夹已经存在，删除并重新创建
    if DST_DIR.exists():
        print(f"⚠️ 目标目录 {DST_DIR} 已存在，正在删除并重新创建...")
        shutil.rmtree(DST_DIR)

    # 创建目标目录
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # 扫描文件夹结构并打印
    for raw_label in os.listdir(SRC_DIR):
        class_dir = SRC_DIR / raw_label
        if class_dir.is_dir():
            if INCLUDE_FOLDERS and raw_label not in INCLUDE_FOLDERS:
                continue  # 跳过不在 INCLUDE_FOLDERS 列表中的文件夹
            if raw_label in IGNORE_FOLDERS:
                print(f"⚠️ 忽略文件夹: {raw_label}")
                continue  # 跳过在 IGNORE_FOLDERS 列表中的文件夹

            image_files = list(class_dir.glob("*.*"))  # 查找所有图像文件
            print(f"📂 文件夹 {raw_label}: 找到 {len(image_files)} 张图像")
            total_files += len(image_files)  # 累加图像文件数

    print(f"\n📊 总共扫描到 {total_files} 张图像\n")

    print(f"🎨 开始处理图像...\n")

    # 清除旧日志文件
    with open(LOG_FILE, "w", encoding="utf-8") as fh:
        fh.write("错误日志\n===========\n")

    # 递归处理文件夹
    for raw_label in os.listdir(SRC_DIR):
        class_dir = SRC_DIR / raw_label
        if class_dir.is_dir():
            # 如果指定了仅包含的文件夹，跳过非包含文件夹
            if INCLUDE_FOLDERS and raw_label not in INCLUDE_FOLDERS:
                continue

            # 如果在忽略文件夹中，跳过该文件夹
            if raw_label in IGNORE_FOLDERS:
                print(f"⚠️ 忽略文件夹: {raw_label}")
                continue

            image_files = list(class_dir.glob("*.*"))  # 查找所有图像文件
            for img_path in tqdm(image_files, desc=f"🔄 处理 {raw_label} 类别图像"):
                dst = DST_DIR / raw_label / img_path.name
                dst.parent.mkdir(parents=True, exist_ok=True)  # 创建目标目录

                process_image(img_path, dst)

    print(f"✅ 图像处理完成！")


if __name__ == "__main__":
    process_images()