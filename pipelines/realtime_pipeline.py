#!/usr/bin/env python
"""
realtime_pipeline.py – 两阶段实时 FER（YOLO-v11 修正版）
========================================================
* 支持 USB 摄像头 / 视频文件 / RTSP
* 可选保存输出、可选关闭窗口
"""

from __future__ import annotations
from pathlib import Path
import time
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------
# 项目根目录
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # YOLO-based-Emotion-Recognition/

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
CONFIG = {
    # 训练 run 名（根据你的 checkpoints 目录填写）
    "det_run": "face-yolo11n_20250614-1807482",
    "cls_run": "yolo11n-cls_20250616-1712432",

    # 视频源：0 = 默认摄像头；或 "video.mp4" / "rtsp://..."
    "source": 2,

    # 输出：空串 = 不保存
    "save_path": "",  # e.g. "runs/infer/out.mp4"

    # 推理设置
    "device": "0",         # "0" | "cpu"
    "img_size": 112,       # 分类器输入尺寸
    "conf_thres": 0.5,     # 检测阈值

    # UI
    "show": True,
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "colors": [
        (255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 0, 255),
        (0, 255, 255), (255, 255, 0), (128, 0, 128)
    ],
    "names": ["angry", "disgust", "fear", "happy",
              "sad", "surprise", "neutral"],
}

# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def draw_label(img: np.ndarray, text: str, tl: tuple[int, int],
               color: tuple[int, int, int]) -> None:
    """
    在 tl(x, y) 位置画半透明文字背景 + 文字
    """
    (w, h), _ = cv2.getTextSize(text, CONFIG["font"], 0.6, 1)
    cv2.rectangle(img, tl, (tl[0] + w + 6, tl[1] - h - 6),
                  color, -1, cv2.LINE_AA)
    cv2.putText(
        img, text, (tl[0] + 3, tl[1] - 3), CONFIG["font"],
        0.6, (255, 255, 255), 1, cv2.LINE_AA
    )

# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    cfg = CONFIG

    # ---------- 模型 ----------
    print("🔹 Loading models ...")
    det_weights = PROJECT_ROOT / "runs" / "det" / "train" \
        / cfg["det_run"] / "weights" / "best.pt"
    cls_weights = PROJECT_ROOT / "runs" / "cls" / "train" \
        / cfg["cls_run"] / "weights" / "best.pt"

    det_model = YOLO(str(det_weights))
    cls_model = YOLO(str(cls_weights))

    # ---------- 视频 ----------
    cap = cv2.VideoCapture(cfg["source"])
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源 {cfg['source']}")

    # ---------- VideoWriter ----------
    writer: cv2.VideoWriter | None = None
    if cfg["save_path"]:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Path(cfg["save_path"]).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(cfg["save_path"], fourcc,
                                 fps_src, (width, height))
        print(f"💾 Saving output to {cfg['save_path']}")

    # ---------- 主循环 ----------
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 人脸检测
        det_res = det_model(
            frame, imgsz=640, conf=cfg["conf_thres"],
            device=cfg["device"], verbose=False
        )[0]

        for box in det_res.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # 分类
            face_resized = cv2.resize(face,
                                      (cfg["img_size"], cfg["img_size"]))
            cls_res = cls_model(
                face_resized, imgsz=cfg["img_size"],
                device=cfg["device"], verbose=False
            )[0]

            idx = int(cls_res.probs.top1)
            prob = float(cls_res.probs.top1conf)
            label = f"{cfg['names'][idx]} {prob:.2f}"

            color = cfg["colors"][idx % len(cfg["colors"])]
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          color, 2, cv2.LINE_AA)
            draw_label(frame, label, (x1, y1), color)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev + 1e-6)
        prev = now
        cv2.putText(frame, f"{fps:.1f} FPS", (10, 30),
                    cfg["font"], 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # 显示 / 保存
        if cfg["show"]:
            cv2.imshow("FER Realtime", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        if writer is not None:
            writer.write(frame)

    # ---------- 资源释放 ----------
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("✅ 结束")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
