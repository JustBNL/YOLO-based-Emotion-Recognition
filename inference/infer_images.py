from pathlib import Path
from ultralytics import YOLO
import sys
import cv2
import numpy as np
import time
from tqdm import tqdm

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
CONFIG = {
    "det_run": "yolo11n_20250617-115322",
    "cls_run": "yolo11s-cls_20250625-212446-new-clean-RFAConv_GAM",
    "input_dir": "data/imgs",
    "output_dir": "output/imgs",
    "device": "0",
    "img_size": 224,
    "conf_thres": 0.5,
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "colors": [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 0, 255),
                (0, 255, 255), (255, 255, 0), (128, 0, 128)],
    "names": ["angry", "disgust", "fear", "happy", "sad" ,"neutral", "surprise" ],
    "label_display": "label",  # or "code"
    "overwrite_output": False  # 覆盖模式
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_models(cfg):
    cfg["det_path"] = PROJECT_ROOT / f"runs/det/train/{cfg['det_run']}/weights/best.pt"
    cfg["cls_path"] = PROJECT_ROOT / f"runs/cls/train/{cfg['cls_run']}/weights/best.pt"
    cfg["input_dir"] = SCRIPT_DIR / cfg["input_dir"]
    cfg["output_dir"] = SCRIPT_DIR / cfg["output_dir"]

    if not cfg["det_path"].exists():
        sys.exit(f"❌ 检测模型文件未找到: {cfg['det_path']}")
    if not cfg["cls_path"].exists():
        sys.exit(f"❌ 分类模型文件未找到: {cfg['cls_path']}")

    try:
        print("🔹 Loading models ...")
        det_model = YOLO(str(cfg["det_path"]))
        cls_model = YOLO(str(cfg["cls_path"]))
        return det_model, cls_model
    except Exception as e:
        sys.exit(f"❌ 模型加载失败: {e}")


def validate_input_dir(input_dir: Path):
    if not input_dir.exists():
        sys.exit(f"❌ 输入图像目录不存在: {input_dir}")
    if not any(input_dir.glob("*.*")):
        sys.exit(f"⚠️ 输入图像目录为空: {input_dir}")


def draw_label(img: np.ndarray, text: str, tl: tuple[int, int], color: tuple[int, int, int]) -> None:
    font_scale = max(img.shape[1] / 800.0, 0.8)
    thickness = max(1, int(font_scale * 2))
    (w, h), _ = cv2.getTextSize(text, CONFIG["font"], font_scale, thickness)
    top_left = tl
    bottom_right = (tl[0] + w + 10, tl[1] - h - 10)
    cv2.rectangle(img, top_left, bottom_right, color, -1, cv2.LINE_AA)
    cv2.putText(img, text, (tl[0] + 5, tl[1] - 5), CONFIG["font"], font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)


def get_unique_filename(base_path: Path) -> Path:
    if CONFIG["overwrite_output"] or not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    i = 1
    while True:
        new_path = base_path.with_name(f"{stem}_{i}{suffix}")
        if not new_path.exists():
            return new_path
        i += 1


def run_inference(cfg, det_model, cls_model):
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    image_paths = list(cfg["input_dir"].glob("*.*"))

    for img_path in tqdm(image_paths, desc="Processing images"):
        start_time = time.time()

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠️ Failed to read image: {img_path.name}")
                continue

            det_res = det_model(img, imgsz=640, conf=cfg["conf_thres"], device=cfg["device"], verbose=False)[0]

            for box in det_res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                face = img[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_resized = cv2.resize(face, (cfg["img_size"], cfg["img_size"]))
                cls_res = cls_model(face_resized, imgsz=cfg["img_size"], device=cfg["device"], verbose=False)[0]

                class_bias = np.array([1.6, 1.4, 1.5, 0.7, 0.7, 1.6, 1.2], dtype=np.float32)

                logits = cls_res.probs.data.cpu().numpy()  # shape: (7,)
                adjusted_logits = logits * class_bias
                probs = adjusted_logits / np.sum(adjusted_logits)  # softmax后可以替代概率

                idx = int(np.argmax(probs))
                prob = float(probs[idx])  # 这是加权后的“伪概率”

                if cfg.get("label_display") == "code":
                    label = f"{idx}"
                else:
                    label = f"{cfg['names'][idx]} {prob:.2f}"

                color = cfg["colors"][idx % len(cfg["colors"])]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                draw_label(img, label, (x1, y1), color)

            out_name = f"{cfg['cls_run']}_{cfg['det_run']}_{img_path.name}"
            out_path = get_unique_filename(cfg["output_dir"] / out_name)
            cv2.imwrite(str(out_path), img)

            elapsed = time.time() - start_time
            print(f"✅ {img_path.name} -> {out_path.relative_to(PROJECT_ROOT)} | ⏱ {elapsed:.2f}s")

        except Exception as e:
            print(f"❌ 处理失败: {img_path.name} | 错误: {e}")

    print("🎉 全部图像处理完成")


def infer_single_image(image: np.ndarray) -> np.ndarray:
    from inference.infer_images import CONFIG, load_models, draw_label
    import cv2
    import numpy as np
    import torch

    det_model, cls_model = load_models(CONFIG)
    img = image.copy()

    det_res = det_model(img, imgsz=640, conf=CONFIG["conf_thres"], device=CONFIG["device"], verbose=False)[0]

    for box in det_res.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_resized = cv2.resize(face, (CONFIG["img_size"], CONFIG["img_size"]))
        cls_res = cls_model(face_resized, imgsz=CONFIG["img_size"], device=CONFIG["device"], verbose=False)[0]

        class_bias = np.array([1.6, 1.4, 1.5, 0.7, 0.7, 1.6, 1.2], dtype=np.float32)
        logits = cls_res.probs.data.cpu().numpy()
        adjusted_logits = logits * class_bias
        probs = adjusted_logits / np.sum(adjusted_logits)
        idx = int(np.argmax(probs))
        label = f"{CONFIG['names'][idx]} {probs[idx]:.2f}"

        color = CONFIG["colors"][idx % len(CONFIG["colors"])]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        draw_label(img, label, (x1, y1), color)

    return img


if __name__ == "__main__":
    det_model, cls_model = load_models(CONFIG)
    validate_input_dir(CONFIG["input_dir"])
    run_inference(CONFIG, det_model, cls_model)
