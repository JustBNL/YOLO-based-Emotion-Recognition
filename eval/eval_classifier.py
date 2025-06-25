#!/usr/bin/env python
"""
eval_classifier.py – 情绪分类模型评估脚本（支持中文控制台输出，图像类型兼容，自动保存报告，BN自动校准）
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
import torch
import torch.nn as nn
from tqdm import tqdm

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
CONFIG = {
    "cls_run": "yolo11s-cls_20250624-232219-clean-cbam-new",
    "data_dir": "datasets/cls/processed/KDEF-2",
    "img_size": 224,
    "device": "0",
    "names": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "report_save": True,
    "report_dir": "eval_result/cls",
    "analyze_classes": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],

    # BN校准相关配置
    "bn_calibration": {
        "enable": False,  # 是否启用BN校准
        "calibration_samples": -1,  # 用于校准的样本数量（-1表示使用全部样本）
        "calibration_batch_size": 64,  # 校准时的批处理大小
        "momentum": 0.1,  # BN层的动量参数
        "eps": 1e-5,  # BN层的epsilon参数
        "verbose": True  # 是否显示校准过程
    }
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
def reset_bn_stats(model: torch.nn.Module) -> None:
    """重置所有BatchNorm层的统计信息"""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.reset_running_stats()


def calibrate_bn_stats(model: torch.nn.Module, calibration_data: list,
                       batch_size: int = 32, device: str = "cpu", verbose: bool = True) -> None:
    """
    使用校准数据重新计算BatchNorm层的运行统计信息

    Args:
        model: 需要校准的模型
        calibration_data: 校准数据列表
        batch_size: 批处理大小
        device: 设备
        verbose: 是否显示进度
    """
    if verbose:
        print("🔧 开始BN校准...")

    # 设置模型为训练模式以更新BN统计信息
    model.train()

    # 先重置所有BN层的统计信息
    reset_bn_stats(model)

    # 批处理校准数据
    num_batches = (len(calibration_data) + batch_size - 1) // batch_size
    progress_bar = tqdm(range(num_batches), desc="BN校准进度") if verbose else range(num_batches)

    with torch.no_grad():
        for batch_idx in progress_bar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(calibration_data))
            batch_data = calibration_data[start_idx:end_idx]

            # 转换为tensor并移到指定设备
            batch_tensor = torch.stack([
                torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                for img in batch_data
            ]).to(device)

            # 前向传播更新BN统计信息
            try:
                # 对于YOLO模型，直接调用模型的backbone部分
                if hasattr(model, 'model'):
                    _ = model.model(batch_tensor)
                else:
                    _ = model(batch_tensor)
            except Exception as e:
                if verbose:
                    print(f"⚠️  批次 {batch_idx} 校准时出现警告: {e}")
                continue

    # 恢复为评估模式
    model.eval()

    if verbose:
        print("✅ BN校准完成")


def apply_bn_calibration(model: YOLO, calibration_images: list, config: dict) -> None:
    """
    对YOLO模型应用BN校准

    Args:
        model: YOLO模型实例
        calibration_images: 用于校准的图像列表
        config: 校准配置
    """
    cal_config = config.get("bn_calibration", {})

    if not cal_config.get("enable", False):
        return

    # 获取校准样本
    num_samples = cal_config.get("calibration_samples", 200)
    if num_samples == -1 or num_samples >= len(calibration_images):
        cal_data = calibration_images
    else:
        # 随机采样
        indices = np.random.choice(len(calibration_images), num_samples, replace=False)
        cal_data = [calibration_images[i] for i in indices]

    if cal_config.get("verbose", True):
        print(f"📊 使用 {len(cal_data)} 个样本进行BN校准")

    # 获取模型的PyTorch模型
    torch_model = model.model
    device = next(torch_model.parameters()).device

    # 执行校准
    calibrate_bn_stats(
        model=torch_model,
        calibration_data=cal_data,
        batch_size=cal_config.get("calibration_batch_size", 32),
        device=device,
        verbose=cal_config.get("verbose", True)
    )


# ----------------------------------------------------------------------
def evaluate() -> None:
    cfg = CONFIG
    model_path = PROJECT_ROOT / "runs/cls/train" / cfg["cls_run"] / "weights" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"❌ 模型文件未找到: {model_path}")

    data_dir_abs = PROJECT_ROOT / cfg["data_dir"] if not Path(cfg["data_dir"]).is_absolute() else Path(cfg["data_dir"])
    rel_data_dir = data_dir_abs.relative_to(PROJECT_ROOT).as_posix().replace("/", "_")

    # 在输出目录名中标记是否使用了BN校准
    bn_suffix = "_bn_cal" if cfg.get("bn_calibration", {}).get("enable", False) else ""
    out_dir = PROJECT_ROOT / cfg["report_dir"] / f"{cfg['cls_run']}__{rel_data_dir}{bn_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    error_img_dir = out_dir / "error_images"
    error_img_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print("🚀 加载模型...")
    model = YOLO(str(model_path))

    # 加载数据
    print("📂 加载评估数据...")
    images, gt_labels, image_names = load_data(cfg["data_dir"])
    print(f"📊 加载了 {len(images)} 张图像")

    # 应用BN校准
    if cfg.get("bn_calibration", {}).get("enable", False):
        apply_bn_calibration(model, images, cfg)

    name2idx = {name: i for i, name in enumerate(cfg["names"])}
    analyze_class_idxs = [name2idx[name] for name in cfg.get("analyze_classes", [])]

    # 开始推理
    print("🔍 开始模型推理...")
    start_time = time.time()
    pred_labels = []
    top3_preds = []
    all_probs = []

    # 添加进度条
    for i, img in enumerate(tqdm(images, desc="推理进度")):
        pred = model(img, imgsz=cfg["img_size"], device=cfg["device"], verbose=False)[0]
        idx = int(pred.probs.top1)
        pred_labels.append(idx)
        probs = pred.probs.data.tolist() if hasattr(pred.probs, 'data') else pred.probs.tolist()
        top3 = sorted(range(len(probs)), key=lambda i: -probs[i])[:3]
        top3_preds.append(top3)
        all_probs.append(probs)

    elapsed = time.time() - start_time
    print(f"⏱️  推理完成，耗时: {elapsed:.2f} 秒")

    print("\n📊 分类报告:")
    report = classification_report(gt_labels, pred_labels, target_names=cfg["names"], digits=4)
    print(report)

    # 计算准确率
    accuracy = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == pred) / len(gt_labels)
    print(f"🎯 总体准确率: {accuracy:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(gt_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cfg["names"])
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax1, cmap="Blues", xticks_rotation=45)
    ax1.set_title("Confusion Matrix")
    fig1.tight_layout()

    # 绘制归一化混淆矩阵
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", xticklabels=cfg["names"], yticklabels=cfg["names"], cmap="Blues")
    ax2.set_title("Normalized Confusion Matrix")
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")
    fig2.tight_layout()

    if cfg.get("report_save", False):
        # 保存配置和结果
        with open(out_dir / "report.txt", "w", encoding="utf-8") as f:
            f.write("模型评估配置:\n")
            for k, v in cfg.items():
                f.write(f"{k}: {v}\n")
            f.write(f"\n总耗时: {elapsed:.2f} 秒\n")
            f.write(f"总体准确率: {accuracy:.4f}\n")
            f.write(f"BN校准状态: {'启用' if cfg.get('bn_calibration', {}).get('enable', False) else '禁用'}\n")
            f.write("\n分类报告:\n")
            f.write(report)

        # 保存图像
        fig1.savefig(out_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        fig2.savefig(out_dir / "confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')

        # 计算每类准确率
        class_counts = np.bincount(gt_labels, minlength=len(cfg["names"]))
        correct = [(np.array(gt_labels) == i).astype(int) & (np.array(pred_labels) == i).astype(int) for i in
                   range(len(cfg["names"]))]
        correct_counts = [int(c.sum()) for c in correct]
        class_acc = [c / total if total > 0 else 0.0 for c, total in zip(correct_counts, class_counts)]

        # 绘制每类准确率
        plt.figure(figsize=(10, 6))
        bars = plt.bar(cfg["names"], class_acc, color='skyblue', alpha=0.7)
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.xlabel("Classes")
        plt.title("Per-Class Accuracy")
        plt.xticks(rotation=45)

        # 在柱状图上添加数值标签
        for bar, acc in zip(bars, class_acc):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(out_dir / "per_class_accuracy.png", dpi=300, bbox_inches='tight')

        # 保存CSV格式的准确率数据
        csv_path = out_dir / "class_accuracy.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Class", "Total", "Correct", "Accuracy"])
            for name, total, corr, acc in zip(cfg["names"], class_counts, correct_counts, class_acc):
                writer.writerow([name, total, corr, f"{acc:.4f}"])

        # 错误分析
        error_counter = defaultdict(int)
        error_detail_path = out_dir / "error_detail.txt"
        with open(error_detail_path, "w", encoding="utf-8") as f:
            f.write("错误样本分析 (包含预测类别):\n")
            f.write(f"BN校准状态: {'启用' if cfg.get('bn_calibration', {}).get('enable', False) else '禁用'}\n\n")

            for i, (true, pred, top3, img_path, probs) in enumerate(
                    zip(gt_labels, pred_labels, top3_preds, image_names, all_probs)):
                if true in analyze_class_idxs and true != pred:
                    rank = top3.index(true) + 1 if true in top3 else "N/A"
                    tag = f"Top{rank}_correct_but_not_top1 [{cfg['names'][true]}]"
                    error_counter[tag] += 1
                    f.write(
                        f"{img_path} -> GT: {cfg['names'][true]} | Pred: {cfg['names'][pred]} | Top3: {[cfg['names'][k] for k in top3]}\n")

                    # 保存错误样本图像
                    img = images[i].copy()
                    prob = probs[pred] if pred < len(probs) else 0.0
                    cv2.putText(img, f"GT:{cfg['names'][true]}", (2, 14), cfg['font'], 0.4, (0, 0, 255), 1)
                    cv2.putText(img, f"Pred:{cfg['names'][pred]}", (2, 28), cfg['font'], 0.4, (255, 0, 0), 1)
                    cv2.putText(img, f"Conf:{prob:.2f}", (2, 42), cfg['font'], 0.4, (0, 128, 0), 1)
                    cv2.imwrite(str(error_img_dir / f"{Path(img_path).stem}_err.jpg"), img)

        # 保存错误统计
        with open(out_dir / "error_analysis.txt", "w", encoding="utf-8") as f:
            f.write("Top-k 错误分析 (仅分析目标类别):\n")
            f.write(f"BN校准状态: {'启用' if cfg.get('bn_calibration', {}).get('enable', False) else '禁用'}\n\n")
            for k in sorted(error_counter):
                f.write(f"{k}: {error_counter[k]}\n")

        print(f"✅ 报告保存于: {out_dir.resolve()}")

        # 输出BN校准信息
        if cfg.get("bn_calibration", {}).get("enable", False):
            cal_config = cfg["bn_calibration"]
            print(f"🔧 BN校准已启用:")
            print(f"   - 校准样本数: {cal_config.get('calibration_samples', 200)}")
            print(f"   - 批处理大小: {cal_config.get('calibration_batch_size', 32)}")
    else:
        plt.show()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    evaluate()