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
    "data": "D:\\Document\\PycharmProjects\\YOLO-based-Emotion-Recognition\\datasets\\cls\\processed\\affectnet\\images", #YOLObug无法使用yaml文件
    "model": str(SCRIPT_DIR / "yolo11s-cls.pt"),
    "epochs": 150,
    "imgsz": 224,
    "cache": "disk",
    "batch": 32,
    "device": "0",
    "amp": True,
    # "workers": 4,
    "patience": 20,            # 早停
    # "optimizer": "AdamW",

    # 日志与项目
    "use_wandb": False,
    "project_root": str(PROJECT_ROOT / "runs" / "cls" / "train"),
    "log_root": str(PROJECT_ROOT / "runs" / "cls" / "log"),
    "run_name": "",           # 空→自动；或旧 run 目录名继续训练
    "resume": False,
}

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def build_run_name(base: Optional[str], model_path: Path) -> str:
    if base:
        return base
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f"{model_path.stem}_{stamp}"


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

    train_root = Path(cfg["project_root"]).expanduser(); train_root.mkdir(parents=True, exist_ok=True)
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
    run_dir = train_root / run_name

    # 日志设置
    log_file = log_root / f"{run_name}.log"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, 'w', 'utf-8')])
    logger = logging.getLogger("cls_trainer")

    # 捕获 Ultralytics (logging 或 loguru) 输出到日志文件
    from ultralytics.utils import LOGGER
    if hasattr(LOGGER, "add"):
        LOGGER.add(log_file, encoding="utf-8")
    else:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        LOGGER.addHandler(file_handler)

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
    model = YOLO(r"D:\Document\PycharmProjects\YOLO-based-Emotion-Recognition\ultralytics\cfg\models\11\yolo11-cls.yaml").load(cfg["model"])

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
        'patience', 'mixup', 'label_smoothing','cache'}}

    model.train(**train_params,
                project=str(train_root), name=run_name,
                resume=resume_flag, pretrained=True,
                exist_ok=True)

    logger.info(f"✅ 训练完成，总耗时 {fmt_seconds(time.perf_counter() - t0)} | 结果目录 {run_dir}")


if __name__ == '__main__':
    main()
