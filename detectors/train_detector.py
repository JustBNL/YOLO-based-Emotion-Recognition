#!/usr/bin/env python
"""
train_detector.py – v3
======================
YOLO‑v11 人脸检测脚本（无命令行）
---------------------------------
✨ **新增功能**
1. 🔄 *自动续训*：当 `CONFIG["resume"] = True` 且 `run_name` 为空，脚本会在 `runs/train/` 中挑选**时间戳最新**且存在 `weights/last.pt` 的实验继续训练。
2. 🗂 **集中日志**：所有日志写入 `runs/log/<run_name>.log`，不再混到 train 目录。
3. 🛑 **早停**：暴露 Ultralytics 的 `patience` 参数，`patience` 轮内验证集 mAP/Loss 无提升即自动停止训练（默认 20）。

```bash
python detectors/train_detector.py  # 直接运行
```
"""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:
    from ultralytics import YOLO
except ImportError as exc:
    sys.stderr.write("❌ Ultralytics 未安装，请执行 `pip install ultralytics>=0.4`\n")
    raise exc

# ---------------------------------------------------------------------------
# 路径解析
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # detectors/
PROJECT_ROOT = SCRIPT_DIR.parent                      # 项目根目录
DATA_YAML = SCRIPT_DIR / "configs" / "yolo_face_det.yaml"
if not DATA_YAML.exists():
    sys.exit(f"❌ 找不到数据配置文件 {DATA_YAML}，请检查路径！")

# ---------------------------------------------------------------------------
# 用户配置
# ---------------------------------------------------------------------------
CONFIG: dict = {
    "data": str(DATA_YAML),
    "model": str(SCRIPT_DIR / "yolo11n.pt"),
    "epochs": 1,
    "imgsz": 640,
    "batch": -1,
    "device": "0",
    "cache": "disk",
    "freeze": 0,
    "amp": True,
    "workers": 4,
    "patience": 20,            # 早停：验证 mAP/Loss patience 轮无提升即停止

    # 日志与项目
    "use_wandb": False,
    "project_root": str(PROJECT_ROOT / "runs" / "train"),
    "log_root": str(PROJECT_ROOT / "runs" / "log"),
    "run_name": "",           # 空→自动；或填旧 run 目录名
    "resume": False,           # True 自动续训
}

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def build_run_name(base: Optional[str], model_path: Path) -> str:
    if base:
        return base
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f"face-{model_path.stem}_{stamp}"


def fmt_seconds(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))


def find_latest_run(train_root: Path) -> Optional[Path]:
    """返回 runs/train 下最新修改且包含 weights/last.pt 的目录。"""
    candidates = [p for p in train_root.iterdir() if p.is_dir() and (p / 'weights/last.pt').exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


class EpochTimer:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.t0 = 0.0

    def on_train_epoch_start(self, _):
        self.t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer):
        self.logger.info(f"⏱️ Epoch {trainer.epoch + 1}/{trainer.args.epochs} 用时: {fmt_seconds(time.perf_counter() - self.t0)}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = CONFIG.copy()

    train_root = Path(cfg["project_root"]).expanduser()
    log_root = Path(cfg["log_root"]).expanduser()
    log_root.mkdir(parents=True, exist_ok=True)

    # ---------------- 选择 run 目录 ----------------
    if cfg["resume"] and not cfg["run_name"]:
        latest = find_latest_run(train_root)
        if latest:
            cfg["run_name"] = latest.name
        else:
            print("⚠️ 未找到可续训的实验，将启动新训练。")
            cfg["resume"] = False

    run_name = build_run_name(cfg["run_name"], Path(cfg["model"]))
    run_dir = train_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 日志 ----------------
    log_file = log_root / f"{run_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, 'w', encoding='utf-8')],
    )
    logger = logging.getLogger("trainer")

    # ---------------- Resume 逻辑 ----------------
    resume_flag = False
    if cfg["resume"]:
        ckpt = run_dir / 'weights' / 'last.pt'
        if ckpt.exists():
            cfg["model"] = str(ckpt)
            resume_flag = True
            logger.info(f"🔄 继续训练 {ckpt}")
        else:
            logger.warning("Resume=True 但未找到 last.pt，改为新训练。")

    # ---------------- 初始化模型 ----------------
    model = YOLO(cfg["model"])

    # 回调注册
    etimer = EpochTimer(logger)
    model.add_callback('on_train_epoch_start', etimer.on_train_epoch_start)
    model.add_callback('on_train_epoch_end', etimer.on_train_epoch_end)

    if cfg.get("use_wandb"):
        from ultralytics.utils.callbacks.wb import WandbCallback
        model.add_callback('on_fit_start', WandbCallback())
        logger.info("📊 W&B 已启用 (需 `yolo settings wandb=True`)")

    # ---------------- 开始训练 ----------------
    t0 = time.perf_counter()
    logger.info(f"🚀 开始训练: {run_name}")

    train_params = {k: v for k, v in cfg.items() if k in {
        'data', 'epochs', 'imgsz', 'batch', 'device', 'cache', 'freeze', 'amp', 'workers', 'patience'} }

    model.train(
        **train_params,
        project=str(train_root),
        name=run_name,
        resume=resume_flag,
        pretrained=True,
    )

    logger.info(f"✅ 训练完成，总耗时: {fmt_seconds(time.perf_counter() - t0)} | 结果目录: {run_dir}")


if __name__ == '__main__':
    main()