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
    "epochs": 150,
    "imgsz": 1024,
    "batch": 2,
    "device": "0",
    "cache": "disk",
    "freeze": 0,
    "amp": True,
    # "workers": 2,
    "patience": 20,            # 早停：验证 mAP/Loss patience 轮无提升即停止

    # 日志与项目
    "use_wandb": False,
    "project_root": str(PROJECT_ROOT / "runs" / "det" / "train"),
    "log_root": str(PROJECT_ROOT / "runs" / "det" / "log"),
    "run_name": "",           # 空→自动；或填旧 run 目录名
    "resume": True,           # True 自动续训

    # "lr0":           0.01,   # 初始学习率（SGD） :contentReference[oaicite:6]{index=6}
    # "lrf":           0.10,   # 余弦衰减终值
    # "momentum":      0.937,
    # "weight_decay":  0.0005,
    # "warmup_epochs": 3,
    #
    # # 色彩增强
    # "hsv_h": 0.015,
    # "hsv_s": 0.70,
    # "hsv_v": 0.40,
    #
    # # 几何增强
    # "fliplr": 0.50,
    # "flipud": 0.0,
    # "degrees": 0.0,
    # "shear": 0.0,
    # "perspective": 0.0,
    #
    # # Mix/Mosaic 增强
    # "mosaic":     0.80,
    # "mixup":      0.0,
    # "copy_paste": 0.0,
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


def find_latest_run(train_root: Path) -> Optional[Path]:
    runs = [p for p in train_root.iterdir() if p.is_dir() and (p / 'weights/last.pt').exists()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


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

    # ---------------- 日志 ----------------
    log_file = log_root / f"{run_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, 'w', encoding='utf-8')],
    )
    logger = logging.getLogger("trainer")

    # 捕获 Ultralytics (logging 或 loguru) 输出到日志文件
    from ultralytics.utils import LOGGER
    if hasattr(LOGGER, "add"):
        # loguru 风格 (旧版)
        LOGGER.add(log_file, encoding="utf-8")
    else:
        # logging.Logger (Ultralytics >= 8.2)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        LOGGER.addHandler(file_handler)

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
        logger.info("📊 已启用 W&B 日志")

    # ---------------- 开始训练 ----------------
    t0 = time.perf_counter(); logger.info(f"🚀 开始训练检测器: {run_name}")
    train_params = {k: v for k, v in cfg.items() if k in {
        # ——核心训练参数——
        "data", "model", "epochs", "imgsz", "batch", "device",
        "cache", "freeze", "amp", "workers", "patience",

        # ——超参数——
        "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
        "hsv_h", "hsv_s", "hsv_v",
        "fliplr", "flipud", "degrees", "shear", "perspective",
        "mosaic", "mixup", "copy_paste",} }

    model.train(
        **train_params,
        project=str(train_root),
        name=run_name,
        resume=resume_flag,
        pretrained=True,
        exist_ok=True,
    )

    logger.info(f"✅ 训练完成，总耗时: {fmt_seconds(time.perf_counter() - t0)} | 结果目录: {run_dir}")


if __name__ == '__main__':
    main()