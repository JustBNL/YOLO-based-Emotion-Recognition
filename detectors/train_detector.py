#!/usr/bin/env python
"""
train_detector.py
=================
🚀 **YOLO‑v11 人脸检测微调脚本（无命令行）**

直接运行即可：
```bash
python detectors/train_detector.py         # 从项目根运行
# 或
python train_detector.py                    # 在 detectors 目录运行也 OK
```
> 绝对路径自动解析：无论当前工作目录在哪，脚本都会找到 `configs/yolo_face.yaml`。

---
⚠️ **Ultralytics 版本提示**
* `fp16` 参数已更名 **`amp`** (Automatic Mixed Precision)
* 若想启用 Weights & Biases：
  1. 终端执行 `yolo settings wandb=True`（只需一次）；
  2. 把 `CONFIG["use_wandb"] = True`。

---
功能概要
* 🕒 自动命名输出目录
* 🔄 断点续训
* 📊 (可选)W&B 日志
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as exc:
    sys.stderr.write("❌ Ultralytics 未安装，请执行 `pip install ultralytics>=0.4`\n")
    raise exc

# ---------------------------------------------------------------------------
# 路径解析：确保 DATA YAML 能在任何工作目录下被找到
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # detectors/
PROJECT_ROOT = SCRIPT_DIR.parent                      # 项目根目录
DATA_YAML = SCRIPT_DIR / "configs" / "yolo_face.yaml"  # 绝对路径

if not DATA_YAML.exists():
    sys.exit(f"❌ 找不到数据配置文件 {DATA_YAML}，请检查路径！")

# ====================== 👇 用户配置区域 👇 ======================
CONFIG: dict = {
    # 数据与模型
    "data": str(DATA_YAML),         # 已解析为绝对路径
    "model": str(PROJECT_ROOT / "yolo11n.pt"),  # 预训练权重或 ckpt

    # 训练超参
    "epochs": 10,
    "imgsz": 640,
    "batch": 16,
    "device": "0",                # GPU 索引；CPU 请设为 ""
    "cache": "disk",
    "freeze": 0,
    "amp": False,                  # 混合精度训练
    "workers": 4,

    # 日志
    "use_wandb": False,

    # 项目管理
    "project_root": str(PROJECT_ROOT / "runs" / "train"),
    "run_name": "",               # 留空自动命名
    "resume": False,               # True ➡ 断点续训
}
# ==============================================================


def _build_run_name(base: str | None, model_path: str | Path) -> str:
    """生成形如 `face-yolo11n_20250615-142530` 的目录名。"""
    if base:
        return base
    stem = Path(model_path).stem  # e.g. yolo11n
    return f"face-{stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def main() -> None:
    cfg = CONFIG  # shorthand

    project_dir = Path(cfg["project_root"]).expanduser()
    run_name = _build_run_name(cfg["run_name"], cfg["model"])
    run_dir = project_dir / run_name

    # 🔄 断点续训逻辑
    resume_flag: str | bool = False
    if cfg["resume"]:
        ckpt = run_dir / "weights/last.pt"
        if ckpt.exists():
            resume_flag = ckpt.as_posix()
            print(f"🔄 正在从 {ckpt} 恢复训练…")
        else:
            print("⚠️  启用了恢复，但未找到 checkpoint，改为全新训练。")

    # 初始化 YOLO 模型
    model = YOLO(cfg["model"])

    print(f"🚀 开始训练: {run_name}")

    # 提取 YOLO.train 支持的关键字
    train_keys = {
        "data", "epochs", "imgsz", "batch", "device", "cache", "freeze", "amp", "workers"
    }
    train_kwargs = {k: v for k, v in cfg.items() if k in train_keys}

    # 🔗 W&B 回调（可选）
    if cfg.get("use_wandb"):
        from ultralytics.utils.callbacks.wb import WandbCallback
        model.add_callback("on_fit_start", WandbCallback())
        print("📊 W&B 回调已注册 (确保执行了 `yolo settings wandb=True`)\n")

    # 训练
    model.train(
        **train_kwargs,
        project=project_dir.as_posix(),
        name=run_name,
        resume=resume_flag,
        pretrained=True,
    )

    print(f"✅ 训练完成，文件已保存至 {run_dir}")


if __name__ == "__main__":
    main()
