#!/usr/bin/env python

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
SCRIPT_DIR = Path(__file__).resolve().parent          # classifiers/
PROJECT_ROOT = SCRIPT_DIR.parent                      # 项目根目录

# ---------------------------------------------------------------------------
# 用户配置
# ---------------------------------------------------------------------------
CONFIG: dict = {
    # 数据与模型
    "data": "D:\\Document\\PycharmProjects\\YOLO-based-Emotion-Recognition\\datasets\\processed\\fer2013\\images", #YOLObug无法使用yaml文件
    "model": str(SCRIPT_DIR / "yolo11n-cls.pt"),
    "epochs": 10,
    "imgsz": 112,
    "mixup": 0.2,           # 0 关闭
    "label_smoothing": 0.1,
    "batch": -1,            # 自动 batch
    "device": "0",
    "amp": True,
    "workers": 4,
    "patience": 5,            # 早停

    # 日志与项目
    "use_wandb": False,
    "project_root": str(PROJECT_ROOT / "runs" / "cls_train"),
    "log_root": str(PROJECT_ROOT / "runs" / "log"),
    "run_name": "",           # 空→自动；或旧 run 目录名继续训练
    "resume": False,           # True 自动续训
}

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def build_run_name(base: Optional[str], model_path: Path) -> str:
    if base:
        return base
    return f"fer-{model_path.stem}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def fmt_seconds(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))


def find_latest_run(root: Path) -> Optional[Path]:
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / 'weights/last.pt').exists()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


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
    log_root = Path(cfg["log_root"]).expanduser(); log_root.mkdir(parents=True, exist_ok=True)

    # 自动挑选最新实验续训
    if cfg["resume"] and not cfg["run_name"]:
        latest = find_latest_run(train_root)
        if latest:
            cfg["run_name"] = latest.name
        else:
            print("⚠️ 未找到可续训分类实验，将重新训练。")
            cfg["resume"] = False

    run_name = build_run_name(cfg["run_name"], Path(cfg["model"]))
    run_dir = train_root / run_name; run_dir.mkdir(parents=True, exist_ok=True)

    # 日志设置
    log_file = log_root / f"{run_name}.log"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, 'w', 'utf-8')])
    logger = logging.getLogger("cls_trainer")

    # Resume 逻辑
    resume_flag = False
    if cfg["resume"]:
        ckpt = run_dir / 'weights' / 'last.pt'
        if ckpt.exists():
            cfg["model"] = str(ckpt)
            resume_flag = True
            logger.info(f"🔄 继续训练 {ckpt}")
        else:
            logger.warning("Resume=True 但未找到 last.pt，改为新训练。")

    # 初始化模型（Ultralytics 分类任务会自动识别）
    model = YOLO(cfg["model"])

    # 回调
    etimer = EpochTimer(logger)
    model.add_callback('on_train_epoch_start', etimer.on_train_epoch_start)
    model.add_callback('on_train_epoch_end', etimer.on_train_epoch_end)
    if cfg.get("use_wandb"):
        from ultralytics.utils.callbacks.wb import WandbCallback
        model.add_callback('on_fit_start', WandbCallback())
        logger.info("📊 已启用 W&B 日志")

    # 开始训练
    t0 = time.perf_counter(); logger.info(f"🚀 开始训练分类器: {run_name}")
    train_params = {k: v for k, v in cfg.items() if k in {
        'data', 'epochs', 'imgsz', 'batch', 'device', 'amp', 'workers',
        'patience', 'mixup', 'label_smoothing'}}

    model.train(**train_params,
                project=str(train_root), name=run_name,
                resume=resume_flag, pretrained=True)

    logger.info(f"✅ 训练完成，总耗时 {fmt_seconds(time.perf_counter() - t0)} | 结果目录 {run_dir}")


if __name__ == '__main__':
    main()
