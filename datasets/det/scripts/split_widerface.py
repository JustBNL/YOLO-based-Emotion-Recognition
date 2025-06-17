# split_widerface.py
"""
把 wider-face-for-yolo-training 拆分成 train / val / test (8:1:1)
目录结果：
images/train  images/val  images/test
labels/train  labels/val  labels/test
"""

import random
import shutil
from pathlib import Path

root = Path("../processed/wider-face-for-yolo-training").resolve()
img_dir = root / "images"
lbl_dir = root / "labels"

imgs = list(img_dir.glob("*.jpg"))
random.seed(0)

for img in imgs:
    r = random.random()
    if r < 0.8:
        split = "train"
    elif r < 0.9:
        split = "val"
    else:
        split = "test"

    # 目标路径
    dest_img = img_dir / split / img.name
    dest_lbl = lbl_dir / split / f"{img.stem}.txt"

    # 创建子目录
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    dest_lbl.parent.mkdir(parents=True, exist_ok=True)

    # 移动图片和标签
    shutil.move(str(img), str(dest_img))
    shutil.move(str(lbl_dir / f"{img.stem}.txt"), str(dest_lbl))

print("✅ 数据集拆分完成！")
