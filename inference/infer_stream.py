from __future__ import annotations

from pathlib import Path
from threading import Thread
from queue import Queue, Empty
from typing import List, Union, Tuple, Any
import time
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# ----------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    "det_run": "face-yolo11n_20250614-1807482",
    "cls_run": "yolo11s-cls_20250625-201726-new-clean-RFAConv",

    # 输入 / 输出
    "input": 2,                       # 路径 | 目录 | int(摄像头) | URL
    "output_dir": "output/videos",    # 输出视频目录

    # 推理参数
    "device": "cuda",                 # "cpu" | "0" | "0,1"  # 改为 cpu 更安全
    "img_size": 224,                  # 分类模型输入尺寸 (正方形)
    "conf_thres": 0.5,
    "half": False,                    # 默认关闭半精度，避免兼容性问题

    # 运行时选项
    "display": True,                  # 是否显示窗口
    "save_video": True,
    "overwrite_output": False,
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "colors": [                       # 绘制框颜色循环
        (255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 0, 255),
        (0, 255, 255), (255, 255, 0), (128, 0, 128),
    ],
    "names": ["angry", "disgust", "fear", "happy", "sad", "neutral", "surprise"],
    "label_display": "label",        # "label" | "code"

    # FPS 调优
    "skip_frames": 1,                 # 静态跳帧：每 N 帧处理一帧
    "auto_skip": True,                # 动态调节跳帧
    "target_fps": 30,                # 目标实时 FPS

    # 线程化 I/O
    "queue_size": 16,                # 预读取缓冲帧数
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 提高 matmul 精度（Transformer 模型常用）
torch.set_float32_matmul_precision("high")

# ----------------------------------------------------------------------
# 工具类
# ----------------------------------------------------------------------

class FrameGrabber(Thread):
    """后台连续读取视频帧，避免推理线程被 I/O 阻塞。"""

    def __init__(self, src: Union[str, int], queue_size: int = 16):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 无法打开视频源: {src}")
        self.q: Queue[np.ndarray] = Queue(maxsize=queue_size)
        self.stopped = False

    def run(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    break
                if not self.q.full():  # 再次检查，避免竞态条件
                    try:
                        self.q.put(frame, timeout=0.1)
                    except:
                        pass
            else:
                time.sleep(0.001)  # 让出 CPU

    def read(self, timeout: float = 1.0) -> Tuple[bool, np.ndarray | None]:
        try:
            frame = self.q.get(timeout=timeout)
            return True, frame
        except Empty:
            return False, None

    def more(self) -> bool:
        return not self.q.empty() or not self.stopped

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()


# ----------------------------------------------------------------------
# 模型加载
# ----------------------------------------------------------------------

def load_models(cfg: dict) -> tuple[YOLO, YOLO]:
    """加载检测与分类模型。"""
    cfg["det_path"] = PROJECT_ROOT / f"runs/det/train/{cfg['det_run']}/weights/best.pt"
    cfg["cls_path"] = PROJECT_ROOT / f"runs/cls/train/{cfg['cls_run']}/weights/best.pt"

    for key in ("det_path", "cls_path"):
        if not cfg[key].exists():
            print(f"⚠️ 模型文件未找到: {cfg[key]}")
            print("尝试使用默认 YOLO 模型...")
            # 使用默认模型作为后备
            if key == "det_path":
                cfg[key] = "yolo11n.pt"  # 人脸检测用默认模型
            else:
                cfg[key] = "yolo11s-cls.pt"  # 分类用默认模型

    print("🔹 正在加载模型 …")
    try:
        det = YOLO(str(cfg["det_path"]))
        cls = YOLO(str(cfg["cls_path"]))
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("使用默认模型...")
        det = YOLO("yolo11n.pt")
        cls = YOLO("yolo11s-cls.pt")

    device = cfg["device"]
    det.to(device)
    cls.to(device)

    if cfg["half"] and torch.cuda.is_available() and device != "cpu":
        try:
            det.model.half()
            cls.model.half()
        except:
            print("⚠️ 半精度转换失败，使用全精度")
            cfg["half"] = False

    return det, cls


# ----------------------------------------------------------------------
# 视频源解析
# ----------------------------------------------------------------------

def get_video_sources(src: Union[str, int, Path]) -> List[Union[int, str]]:
    """根据输入返回视频源列表。"""
    if isinstance(src, int):
        return [src]
    src_path = Path(str(src))
    if src_path.exists():
        if src_path.is_file():
            return [str(src_path)]
        if src_path.is_dir():
            vids = sorted([p for p in src_path.iterdir() if p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}])
            if not vids:
                print(f"⚠️ 目录为空: {src_path}")
                return []
            return [str(v) for v in vids]
    # 其余情况按 URL/未知路径处理
    return [str(src)]


# ----------------------------------------------------------------------
# 绘制辅助
# ----------------------------------------------------------------------

def draw_label(img: np.ndarray, text: str, tl: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    """在目标框左上绘制带背景的文本。"""
    font_scale = max(img.shape[1] / 800.0, 0.6)
    thickness = max(1, int(font_scale * 2))
    (w, h), _ = cv2.getTextSize(text, CONFIG["font"], font_scale, thickness)
    top_left = tl
    bottom_right = (tl[0] + w + 10, tl[1] - h - 10)
    cv2.rectangle(img, top_left, bottom_right, color, -1, cv2.LINE_AA)
    cv2.putText(img, text, (tl[0] + 5, tl[1] - 5), CONFIG["font"], font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)


# ----------------------------------------------------------------------
# 视频写入
# ----------------------------------------------------------------------

def unique_path(path: Path) -> Path:
    """若文件已存在则自动追加索引避免覆盖。"""
    if CONFIG["overwrite_output"] or not path.exists():
        return path
    stem, suf = path.stem, path.suffix
    for i in range(1, 1000):
        cand = path.with_name(f"{stem}_{i}{suf}")
        if not cand.exists():
            return cand
    return path


def create_writer(cfg: dict, source: Union[str, int], frame: np.ndarray) -> cv2.VideoWriter | None:
    """根据首帧分辨率创建 VideoWriter。"""
    if not cfg["save_video"]:
        return None
    try:
        out_dir = SCRIPT_DIR / cfg["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        name = f"camera_{source}.mp4" if isinstance(source, int) else f"{Path(str(source)).stem}.mp4"
        out_path = unique_path(out_dir / name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(str(out_path), fourcc, cfg["target_fps"], (w, h))
        return writer
    except Exception as e:
        print(f"⚠️ 创建视频写入器失败: {e}")
        return None


# ----------------------------------------------------------------------
# 推理主循环
# ----------------------------------------------------------------------

def process_stream(cfg: dict, source: Union[int, str], det: YOLO, cls: YOLO) -> None:
    try:
        grabber = FrameGrabber(source, queue_size=cfg["queue_size"])
        grabber.start()
    except Exception as e:
        print(f"❌ 无法启动帧采集器: {e}")
        return

    # 初始化计时
    prev_time = time.time()
    avg_fps = 0.0
    frame_count = 0

    # 等待首帧以确定分辨率
    ok, frame = grabber.read()
    if not ok:
        print("⚠️ 无法读取首帧，跳过该源。")
        grabber.stop()
        return

    writer = create_writer(cfg, source, frame)

    pbar = tqdm(total=0, position=0, unit="fps", bar_format="{desc}")

    try:
        while grabber.more():
            ok, frame = grabber.read()
            if not ok:
                break

            frame_count += 1
            # 静态跳帧
            if cfg["skip_frames"] > 1 and frame_count % cfg["skip_frames"] != 0:
                continue

            # ------------------------------------
            # 1. 人脸检测
            # ------------------------------------
            try:
                det_res = det.predict(frame, conf=cfg["conf_thres"], device=cfg["device"], verbose=False)[0]
                boxes = det_res.boxes.xyxy.cpu().numpy() if det_res.boxes is not None else []
            except Exception as e:
                print(f"⚠️ 检测失败: {e}")
                boxes = []

            faces = []
            coords = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                if x2 <= x1 or y2 <= y1:  # 无效框
                    continue
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                faces.append(face)
                coords.append((x1, y1, x2, y2))

            # ------------------------------------
            # 2. 批量表情分类
            # ------------------------------------
            labels = []
            if faces:
                try:
                    batch = []
                    for f in faces:
                        img = cv2.resize(f, (cfg["img_size"], cfg["img_size"]))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img.transpose(2, 0, 1) / 255.0
                        batch.append(img)
                    batch_tensor = torch.from_numpy(np.stack(batch)).float()
                    if cfg["half"] and torch.cuda.is_available() and cfg["device"] != "cpu":
                        batch_tensor = batch_tensor.half()
                    batch_tensor = batch_tensor.to(cfg["device"])

                    cls_res = cls.predict(batch_tensor, device=cfg["device"], verbose=False)
                    # 类别权重（你可以调整这些值）
                    class_bias = torch.tensor([1.6, 1.7, 1.5, 0.9, 0.2, 2.8, 5.0], device=cfg["device"])

                    for res in cls_res:
                        prob = res.probs
                        if prob is None:
                            labels.append("?")
                            continue

                        logits = prob.data  # shape: (num_classes,)
                        if logits is None or not isinstance(logits, torch.Tensor):
                            labels.append("?")
                            continue

                        # 加权并softmax
                        adjusted_logits = logits * class_bias
                        adjusted_probs = torch.nn.functional.softmax(adjusted_logits, dim=0)
                        idx = int(torch.argmax(adjusted_probs))

                        if 0 <= idx < len(cfg["names"]):
                            label = cfg["names"][idx]
                        else:
                            label = "unknown"
                        labels.append(label)

                except Exception as e:
                    print(f"⚠️ 分类失败: {e}")
                    labels = ["error"] * len(faces)

            # ------------------------------------
            # 3. 绘制结果
            # ------------------------------------
            for (x1, y1, x2, y2), label in zip(coords, labels):
                try:
                    idx = cfg["names"].index(label) if label in cfg["names"] else 0
                    color = cfg["colors"][idx % len(cfg["colors"])]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    draw_label(frame, label, (x1, y1), color)
                except Exception as e:
                    print(f"⚠️ 绘制失败: {e}")

            # ------------------------------------
            # 4. FPS & 输出
            # ------------------------------------
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt > 0 else 0.0
            avg_fps = 0.9 * avg_fps + 0.1 * fps if avg_fps else fps
            pbar.set_description(f"{source} | FPS: {avg_fps:.1f}")

            if writer:
                try:
                    writer.write(frame)
                except:
                    pass
            if cfg["display"]:
                cv2.imshow(str(source), frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 动态调节跳帧
            if cfg["auto_skip"] and avg_fps:
                if avg_fps < cfg["target_fps"] * 0.9:
                    cfg["skip_frames"] = min(cfg["skip_frames"] + 1, 5)
                elif avg_fps > cfg["target_fps"] * 1.1:
                    cfg["skip_frames"] = max(cfg["skip_frames"] - 1, 1)

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"❌ 处理过程中出错: {e}")
    finally:
        # 清理
        grabber.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        pbar.close()


# ----------------------------------------------------------------------
# 高层 API
# ----------------------------------------------------------------------

def run_infer(*, input_source: Union[int, str] | None = None, target_fps: int | None = None) -> None:
    """供外部脚本调用的简易入口。"""
    if input_source is not None:
        CONFIG["input"] = input_source
    if target_fps is not None:
        CONFIG["target_fps"] = target_fps

    try:
        det_model, cls_model = load_models(CONFIG)
        sources = get_video_sources(CONFIG["input"])
        if not sources:
            print("❌ 没有找到有效的视频源")
            return
        for src in sources:
            process_stream(CONFIG, src, det_model, cls_model)
    except Exception as e:
        print(f"❌ 运行失败: {e}")


# ----------------------------------------------------------------------
# CLI 入口
# ----------------------------------------------------------------------

if __name__ == "__main__":
    run_infer()