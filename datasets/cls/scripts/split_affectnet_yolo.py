#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AffectNet → YOLO 数据集拆分脚本
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Final, Iterable

from PIL import Image
from tqdm import tqdm

# ───────────────────────────────────────────────────────────────
# 配置
# ───────────────────────────────────────────────────────────────

LABELS: Final[list[str]] = [    # 目标类别标签列表（YOLO格式）
    "angry",  # 愤怒
    "disgust",  # 厌恶
    "fear",  # 恐惧
    "happy",  # 高兴
    "sad",  # 悲伤
    "surprise",  # 惊讶
    "neutral",  # 中性
]
CLASS_MAPPING: Final[dict[str, str]] = {    # 原始AffectNet标签到标准化标签的映射
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}
SPLIT_RATIOS: Final[tuple[float, float, float]] = (0.8, 0.1, 0.1)   # 数据集拆分比例 (训练集, 验证集, 测试集)
assert abs(sum(SPLIT_RATIOS) - 1.0) < 1e-6, "拆分比例必须相加为 1"   # 验证拆分比例总和是否为1（允许极小误差）
SRC_DIR = Path("../raw/affectnet")  # 源数据目录（原始AffectNet数据集）
DST_DIR = Path("../processed/affectnet/images") # 目标数据目录（处理后的YOLO格式数据集）
LOG_FILE = "error_log.txt"  # 错误日志文件路径
RESIZE_TO: Final[tuple[int, int]] = (224, 224)  # 图像调整大小目标尺寸 (宽度, 高度)
CONVERT_GRAYSCALE: Final[bool] = True   # 是否将图像转换为灰度图
# ───────────────────────────────────────────────────────────────
# 辅助函数
# ───────────────────────────────────────────────────────────────

def _create_dirs() -> None:
    for split in ("train", "val", "test"):
        for label in LABELS:
            (DST_DIR / split / label).mkdir(parents=True, exist_ok=True)


def _iter_images() -> Iterable[tuple[Path, str]]:
    for raw_label in os.listdir(SRC_DIR):
        if raw_label not in CLASS_MAPPING:
            print(f"⚠️ [警告] 未识别的标签文件夹，已跳过: {raw_label}")
            continue

        mapped = CLASS_MAPPING[raw_label]
        class_dir = SRC_DIR / raw_label

        # 使用集合去重，避免重复统计
        image_files = set()

        # 搜索所有可能的图像文件扩展名（包括大小写变体）
        extensions = [
            "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif",
            "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIFF", "*.TIF"
        ]

        for ext in extensions:
            for p in class_dir.glob(ext):
                # 使用绝对路径作为集合的键，确保去重
                image_files.add(p.resolve())

        # 输出该类别的文件统计
        print(f"📊 类别 {raw_label} → {mapped}: {len(image_files)} 张图片")

        for img_path in image_files:
            yield Path(img_path), mapped


def _process(pair: tuple[Path, Path]) -> None:
    src, dst = pair
    try:
        img = Image.open(src)

        # 根据配置进行颜色空间转换
        if CONVERT_GRAYSCALE:
            img = img.convert("L")  # 转换为灰度图
        else:
            img = img.convert("RGB")  # 保持RGB彩色

        # 调整图像尺寸
        img = img.resize(RESIZE_TO)
        img.save(dst)
    except Exception as exc:
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"❌ 处理失败 {src}: {exc}\n")


# ───────────────────────────────────────────────────────────────
# 主流程
# ───────────────────────────────────────────────────────────────

def split_and_convert(seed: int = 42) -> None:
    random.seed(seed)
    _create_dirs()

    # 清空旧日志
    with open(LOG_FILE, "w", encoding="utf-8") as fh:
        fh.write("错误日志\n===========\n")

    print("🔍 正在统计图像文件...")
    all_images = list(_iter_images())

    print(f"\n🖼️ 总计发现 {len(all_images)} 张图片")
    random.shuffle(all_images)

    n_total = len(all_images)
    n_train = int(n_total * SPLIT_RATIOS[0])
    n_val = int(n_total * SPLIT_RATIOS[1])

    split_boundaries = (n_train, n_train + n_val)

    print(f"📦 数据集拆分:")
    print(f"  🚂 训练集: {n_train} 张 ({SPLIT_RATIOS[0]:.1%})")
    print(f"  🧪 验证集: {n_val} 张 ({SPLIT_RATIOS[1]:.1%})")
    print(f"  🧪 测试集: {n_total - n_train - n_val} 张 ({SPLIT_RATIOS[2]:.1%})")

    tasks: list[tuple[Path, Path]] = []
    for idx, (img_path, label) in enumerate(all_images):
        if idx < split_boundaries[0]:
            split = "train"
        elif idx < split_boundaries[1]:
            split = "val"
        else:
            split = "test"
        dst = DST_DIR / split / label / img_path.name
        tasks.append((img_path, dst))

    # 并行处理图像
    print("\n⚙️ 开始处理图像...")
    print(f"  🔄 图像尺寸调整: {RESIZE_TO[0]}×{RESIZE_TO[1]}")
    print(f"  🎨 颜色空间: {'灰度 (L)' if CONVERT_GRAYSCALE else '彩色 (RGB)'}")

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        futures = [ex.submit(_process, t) for t in tasks]
        for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="🚀 处理图片",
        ):
            pass

    print(f"\n✅ 处理完成！输出目录: {DST_DIR}")

if __name__ == "__main__":
    split_and_convert()