"""
把 wider-face-for-yolo-training 拆分成 train / val / test (8:1:1)
目录结果：
images/train  images/val  images/test
labels/train  labels/val  labels/test
"""

import random
import shutil
from pathlib import Path

# 源数据路径 (raw目录)
source_root = Path("../raw/wider-face-for-yolo-training").resolve()

# 目标路径 (processed目录)
dest_root = Path("../processed/wider-face-for-yolo-training").resolve()

# 源数据子目录
src_img_dir = source_root / "images"
src_lbl_dir = source_root / "labels"

# 目标子目录
dest_img_dir = dest_root / "images"
dest_lbl_dir = dest_root / "labels"

# 获取所有图片文件 (假设都是.jpg)
imgs = list(src_img_dir.glob("*.jpg"))
random.seed(0)  # 固定随机种子确保可复现

for img in imgs:
    # 生成随机分组 (8:1:1)
    r = random.random()
    split = "train" if r < 0.8 else "val" if r < 0.9 else "test"

    # ===== 修正点：操作双路径 =====
    # 源标签路径 (raw目录)
    src_label = src_lbl_dir / f"{img.stem}.txt"

    # 目标路径 (processed目录)
    dest_img = dest_img_dir / split / img.name
    dest_label = dest_lbl_dir / split / f"{img.stem}.txt"

    # 创建目标目录 (processed下)
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    dest_label.parent.mkdir(parents=True, exist_ok=True)

    # 移动文件 (从raw到processed)
    shutil.move(str(img), str(dest_img))
    if src_label.exists():  # 确保标签存在
        shutil.move(str(src_label), str(dest_label))

print("✅ 数据集拆分完成!")