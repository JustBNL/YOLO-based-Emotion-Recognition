#!/usr/bin/env python
"""
eval_classifier.py – 情绪分类模型评估脚本（支持中文控制台输出，图像类型兼容，自动保存报告）
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import time
from collections import Counter, defaultdict

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
CONFIG = {
    "cls_run": "yolo11n-cls_20250616-1712432",
    "data_dir": "datasets/processed/fer2013/images/val",
    "img_size": 112,
    "device": "0",
    "names": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "report_save": True,
    "report_dir": "eval_result",
    "analyze_classes": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]  # 可选: 仅分析特定类别（按名称）
}

# ----------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ----------------------------------------------------------------------
def load_data(data_dir: Path) -> tuple[list[np.ndarray], list[int], list[str]]:
    images, labels, image_names = [], [], []
    data_dir = PROJECT_ROOT / data_dir if not Path(data_dir).is_absolute() else Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"❌ 评估数据路径不存在: {data_dir}")

    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    for idx, class_dir in enumerate(class_dirs):
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            for img_path in class_dir.rglob(ext):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.resize(img, (CONFIG["img_size"], CONFIG["img_size"]))
                images.append(img)
                labels.append(idx)
                image_names.append(str(img_path))

    return images, labels, image_names


# ----------------------------------------------------------------------
def evaluate() -> None:
    cfg = CONFIG
    model_path = PROJECT_ROOT / "runs/cls/train" / cfg["cls_run"] / "weights" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"❌ 模型文件未找到: {model_path}")

    data_dir_abs = PROJECT_ROOT / cfg["data_dir"] if not Path(cfg["data_dir"]).is_absolute() else Path(cfg["data_dir"])
    rel_data_dir = data_dir_abs.relative_to(PROJECT_ROOT).as_posix().replace("/", "_")
    out_dir = PROJECT_ROOT / cfg["report_dir"] / f"{cfg['cls_run']}__{rel_data_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)
    error_img_dir = out_dir / "error_images"
    error_img_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    images, gt_labels, image_names = load_data(cfg["data_dir"])

    name2idx = {name: i for i, name in enumerate(cfg["names"])}
    analyze_class_idxs = [name2idx[name] for name in cfg.get("analyze_classes", [])]

    start_time = time.time()
    pred_labels = []
    top3_preds = []
    all_probs = []
    for img in images:
        pred = model(img, imgsz=cfg["img_size"], device=cfg["device"], verbose=False)[0]
        idx = int(pred.probs.top1)
        pred_labels.append(idx)
        probs = pred.probs.data.tolist() if hasattr(pred.probs, 'data') else pred.probs.tolist()
        top3 = sorted(range(len(probs)), key=lambda i: -probs[i])[:3]
        top3_preds.append(top3)
        all_probs.append(probs)
    elapsed = time.time() - start_time

    print("\n📊 分类报告:")
    report = classification_report(gt_labels, pred_labels, target_names=cfg["names"], digits=4)
    print(report)

    cm = confusion_matrix(gt_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cfg["names"])
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax1, cmap="Blues", xticks_rotation=45)
    ax1.set_title("Confusion Matrix")
    fig1.tight_layout()

    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=cfg["names"], yticklabels=cfg["names"], cmap="Blues")
    ax2.set_title("Normalized Confusion Matrix")
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")
    fig2.tight_layout()

    if cfg.get("report_save", False):
        with open(out_dir / "report.txt", "w", encoding="utf-8") as f:
            f.write("模型评估配置:\n")
            for k, v in cfg.items():
                f.write(f"{k}: {v}\n")
            f.write(f"\n总耗时: {elapsed:.2f} 秒\n")
            f.write("\n分类报告:\n")
            f.write(report)
        fig1.savefig(out_dir / "confusion_matrix.png")
        fig2.savefig(out_dir / "confusion_matrix_normalized.png")

        class_counts = np.bincount(gt_labels, minlength=len(cfg["names"]))
        correct = [(np.array(gt_labels) == i).astype(int) & (np.array(pred_labels) == i).astype(int) for i in range(len(cfg["names"]))]
        correct_counts = [int(c.sum()) for c in correct]
        class_acc = [c / total if total > 0 else 0.0 for c, total in zip(correct_counts, class_counts)]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=cfg["names"], y=class_acc)
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Class Accuracy")
        plt.tight_layout()
        plt.savefig(out_dir / "per_class_accuracy.png")

        csv_path = out_dir / "class_accuracy.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Class", "Total", "Correct", "Accuracy"])
            for name, total, corr, acc in zip(cfg["names"], class_counts, correct_counts, class_acc):
                writer.writerow([name, total, corr, f"{acc:.4f}"])

        error_counter = defaultdict(int)
        error_detail_path = out_dir / "error_detail.txt"
        with open(error_detail_path, "w", encoding="utf-8") as f:
            f.write("错误样本分析 (包含预测类别):\n")
            for i, (true, pred, top3, img_path, probs) in enumerate(zip(gt_labels, pred_labels, top3_preds, image_names, all_probs)):
                if true in analyze_class_idxs and true != pred:
                    rank = top3.index(true) + 1 if true in top3 else "N/A"
                    tag = f"Top{rank}_correct_but_not_top1 [{cfg['names'][true]}]"
                    error_counter[tag] += 1
                    f.write(f"{img_path} -> GT: {cfg['names'][true]} | Pred: {cfg['names'][pred]} | Top3: {[cfg['names'][k] for k in top3]}\n")

                    img = images[i].copy()
                    prob = probs[pred] if pred < len(probs) else 0.0
                    cv2.putText(img, f"GT:{cfg['names'][true]}", (2, 14), cfg['font'], 0.4, (0,0,255), 1)
                    cv2.putText(img, f"Pred:{cfg['names'][pred]}", (2, 28), cfg['font'], 0.4, (255,0,0), 1)
                    cv2.putText(img, f"Conf:{prob:.2f}", (2, 42), cfg['font'], 0.4, (0,128,0), 1)
                    cv2.imwrite(str(error_img_dir / f"{Path(img_path).stem}_err.jpg"), img)

        with open(out_dir / "error_analysis.txt", "w", encoding="utf-8") as f:
            f.write("Top-k 错误分析 (仅分析目标类别):\n")
            for k in sorted(error_counter):
                f.write(f"{k}: {error_counter[k]}\n")

        print(f"✅ 报告保存于: {out_dir.resolve()}")
    else:
        plt.show()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    evaluate()