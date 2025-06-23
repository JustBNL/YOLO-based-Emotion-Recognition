from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from ultralytics import YOLO
import yaml
import time
import hashlib
import json
import shutil
import os


def create_yolo_dataset_structure(
        fold_idx: int,
        train_idx: List[int],
        val_idx: List[int],
        img_paths: List[Path],
        labels: List[int],
        label_map: Dict[int, str],
        fold_dir: Path
) -> Path:
    """为YOLO分类创建标准数据集结构"""

    # 创建YOLO分类数据集结构
    dataset_dir = fold_dir / "dataset"
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    # 清理并创建目录
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    # 为每个类别创建目录
    for label_id, label_name in label_map.items():
        (train_dir / label_name).mkdir(parents=True, exist_ok=True)
        (val_dir / label_name).mkdir(parents=True, exist_ok=True)

    # 复制训练集图片
    train_paths = [img_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    for img_path, label in zip(train_paths, train_labels):
        label_name = label_map[label]
        dst_path = train_dir / label_name / img_path.name
        shutil.copy2(img_path, dst_path)

    # 复制验证集图片
    val_paths = [img_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    for img_path, label in zip(val_paths, val_labels):
        label_name = label_map[label]
        dst_path = val_dir / label_name / img_path.name
        shutil.copy2(img_path, dst_path)

    print(f"      📁 创建数据集结构: 训练集 {len(train_idx)} 张, 验证集 {len(val_idx)} 张")

    return dataset_dir


def save_and_train_one_fold(
        fold_idx: int,
        train_idx: List[int],
        val_idx: List[int],
        img_paths: List[Path],
        labels: List[int],
        label_map: Dict[int, str],
        params: Dict,
        dataset_name: str,
        use_cache: bool = True
) -> Path:
    """训练并保存模型权重，支持缓存功能"""
    print(f"      ⏳ 训练 Fold {fold_idx}...")

    # 简化的缓存路径：weights/dataset_name/fold_idx/
    cache_base_dir = Path("weights") / dataset_name
    fold_dir = cache_base_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # 最佳模型权重路径
    best_ckpt = fold_dir / "train" / "weights" / "best.pt"

    # 简化缓存检查：只检查权重文件是否存在
    if use_cache and best_ckpt.exists():
        print(f"      ✅ 使用缓存的 Fold {fold_idx} 模型")
        return best_ckpt

    # 创建YOLO数据集结构
    dataset_dir = create_yolo_dataset_structure(
        fold_idx, train_idx, val_idx, img_paths, labels, label_map, fold_dir
    )

    # YOLO模型训练
    print(f"      🔄 开始训练 Fold {fold_idx}...")
    base_ckpt = "yolo11s-cls.pt"
    model = YOLO(base_ckpt)

    model.train(
        data=str(dataset_dir),  # 传入数据集目录路径
        epochs=params["epochs"],
        imgsz=params["imgsz"],
        batch=params["batch"],
        patience=params["patience"],
        project=str(fold_dir),
        cache=params["cache"],
        amp=params["amp"],
        workers=params["workers"],
        name="",
        exist_ok=True,
        device=params["device"],
    )

    print(f"      ✅ Fold {fold_idx} 训练完成")
    return best_ckpt


def generate_dataset_name(img_paths: List[Path], n_folds: int, data_dir: Path = None) -> str:
    """生成稳定的数据集名称（确保相同数据集每次生成相同名称）"""
    if data_dir:
        # 使用数据集目录名称和文件数量
        dataset_name = data_dir.name
        file_count = len(img_paths)
        return f"{dataset_name}_{file_count}_{n_folds}fold"
    else:
        # 备用方案：使用第一个文件的父目录名称
        if img_paths:
            parent_name = img_paths[0].parent.name
            file_count = len(img_paths)
            return f"{parent_name}_{file_count}_{n_folds}fold"
        else:
            return f"dataset_{n_folds}fold_empty"


def train_kfold_models(
        img_paths: List[Path],
        labels: List[int],
        label_map: Dict[int, str],
        n_folds: int = 5,
        params: dict = {},
        data_dir: Path = None,
        use_cache: bool = True
) -> List[Path]:
    """K-Fold训练模型并返回每折的权重路径，支持缓存功能"""
    print("📦 启动 K 折训练...")

    # 生成稳定的数据集名称
    dataset_name = generate_dataset_name(img_paths, n_folds, data_dir)
    print(f"   数据集标识: {dataset_name}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    weight_paths = []

    # 循环每一折的训练
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(img_paths, labels)):
        print(f"📁 训练 Fold {fold_idx + 1}/{n_folds}:")
        weight_path = save_and_train_one_fold(
            fold_idx, train_idx, val_idx, img_paths, labels, label_map,
            params, dataset_name, use_cache
        )
        weight_paths.append(weight_path)

    print("✅ K 折训练完成")
    return weight_paths


def kfold_predict(
        img_paths: List[Path],
        labels: List[int],
        weight_paths: List[Path],
        n_folds: int,
        batch_size: int,
        conf_thresh: float,
        config: dict,
        data_dir: Path = None,
        use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """使用K折模型预测，支持缓存功能"""
    print("🔍 开始 K 折模型预测...")

    # 使用与训练时相同的数据集名称生成方法
    dataset_name = generate_dataset_name(img_paths, n_folds, data_dir)
    print(f"   数据集标识: {dataset_name}")

    cache_dir = config["CACHE_DIR"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"pred_{dataset_name}.npz"

    # 如果缓存已存在且允许使用缓存，则加载缓存
    if use_cache and cache_path.exists():
        try:
            print("🔄 使用缓存的预测数据")
            data = np.load(cache_path)
            return data["y_true"], data["y_pred"]
        except Exception as e:
            print(f"⚠️ 缓存加载失败: {e}，重新预测")

    all_probs = []  # 存储所有折的预测概率
    all_true = np.array(labels)  # 存储真实标签
    n_classes = len(set(labels))  # 类别数量

    # 循环遍历每个折的权重文件进行预测
    for i, weight_path in enumerate(weight_paths):
        print(f"   使用 Fold {i + 1} 权重进行预测: {weight_path.name}")

        if not weight_path.exists():
            raise FileNotFoundError(f"权重文件不存在: {weight_path}")

        model = YOLO(str(weight_path))

        # 批量预测，获取每个图片的预测结果
        fold_probs = []

        # 分批处理图片
        for batch_start in tqdm(range(0, len(img_paths), batch_size),
                                desc=f"预测 Fold {i + 1}", leave=False):
            batch_end = min(batch_start + batch_size, len(img_paths))
            batch_paths = [str(p) for p in img_paths[batch_start:batch_end]]

            # YOLO预测
            results = model.predict(batch_paths, conf=conf_thresh, verbose=False)

            # 处理预测结果
            for result in results:
                if hasattr(result, 'probs') and result.probs is not None:
                    # 分类任务的概率
                    prob = result.probs.data.cpu().numpy()
                    if len(prob) == n_classes:
                        fold_probs.append(prob)
                    else:
                        # 如果概率维度不匹配，创建默认概率分布
                        default_prob = np.ones(n_classes) / n_classes
                        fold_probs.append(default_prob)
                else:
                    # 如果没有概率信息，创建默认概率分布
                    default_prob = np.ones(n_classes) / n_classes
                    fold_probs.append(default_prob)

        fold_probs = np.array(fold_probs)
        all_probs.append(fold_probs)
        print(f"   Fold {i + 1} 预测完成，形状: {fold_probs.shape}")

    # 检查所有fold的预测结果形状是否一致
    if len(set(probs.shape for probs in all_probs)) > 1:
        print("⚠️ 警告: 不同fold的预测结果形状不一致")
        # 统一形状
        target_shape = all_probs[0].shape
        for i in range(len(all_probs)):
            if all_probs[i].shape != target_shape:
                print(f"   调整 Fold {i + 1} 的预测结果形状")
                all_probs[i] = np.resize(all_probs[i], target_shape)

    # 计算所有折的平均预测概率
    all_probs = np.array(all_probs)
    pred_probs = np.mean(all_probs, axis=0)

    print(f"✅ 集成预测完成，最终形状: {pred_probs.shape}")

    # 保存预测结果到缓存
    if use_cache:
        try:
            print("💾 保存预测结果到缓存")
            np.savez(cache_path, y_true=all_true, y_pred=pred_probs)
            print(f"   缓存保存至: {cache_path}")
        except Exception as e:
            print(f"⚠️ 缓存保存失败: {e}")

    print("✅ K 折预测完成")
    return all_true, pred_probs