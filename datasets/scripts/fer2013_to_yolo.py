#!/usr/bin/env python
"""convert_fer2013.py – v3 (fast)
=================================
将官方 `fer2013.csv` 拆分为 Ultralytics **分类任务**目录，保持
```
datasets/processed/fer2013/images/{train|val|test}/{angry|...|neutral}/*.jpg
```
并消除 `glob()` O(N²) 瓶颈：**文件名直接用行号 idx**，线性 O(N)。

运行：
```bash
python datasets/scripts/convert_fer2013.py   # 读取下方 CONFIG
```
"""
from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 用户配置
# ---------------------------------------------------------------------------
CONFIG: dict = {
    "csv_path": "../raw/fer2013.csv",            # 原始 CSV 路径
    "out_root": "../processed/fer2013",          # 输出根目录
    "rgb": False,                                       # True→灰度转 3 通道
    "workers": os.cpu_count() or 8,                     # 写文件线程
}

# 标签映射
LABELS = {
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "sad",
    "5": "surprise",
    "6": "neutral",
}
SPLIT_MAP = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

def save_image(row: list[str], idx: int, root: Path, rgb: bool) -> None:
    """把一行 CSV 转成 jpg 并保存."""
    emotion, pixels, usage = row
    dst_dir = root / "images" / SPLIT_MAP[usage] / LABELS[emotion]
    dst_dir.mkdir(parents=True, exist_ok=True)

    img = np.fromstring(pixels, sep=" ", dtype=np.uint8).reshape(48, 48)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(str(dst_dir / f"{idx:05d}.jpg"), img)


def main() -> None:
    cfg = CONFIG
    csv_path = Path(cfg["csv_path"]).expanduser()
    out_root = Path(cfg["out_root"]).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    # 读取所有行
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # 跳表头
        rows = list(reader)

    # 并发写图，行号 idx 直接作为文件名
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
        list(tqdm(ex.map(lambda p: save_image(*p, out_root, cfg["rgb"]),
                        ((row, i) for i, row in enumerate(rows))),
                 total=len(rows), desc="Converting"))

    print(f"✅ 转换完成！图片已保存至 {out_root / 'images'}")


if __name__ == "__main__":
    main()