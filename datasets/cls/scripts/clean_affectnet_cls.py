#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标签清洗：支持阈值调整 + 多算法集成
可使用多权重
- CleanLab
- K-Means 聚类异常检测
- Isolation Forest 全局异常检测
- 集成投票决策
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from cleanlab.classification import CleanLearning
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from ultralytics import YOLO
import yaml

# ───────────────────────────────────────────────────────────────
# 配置参数 (含默认值说明)
# ───────────────────────────────────────────────────────────────
SCRIPT_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SCRIPT_DIR.parent.parent.parent

CONFIG: dict = {
    "DATA_DIR": PROJECT_ROOT / "datasets/cls/raw/affectnet",    # 数据集根目录
    "KFOLD": 5,  # K折交叉验证折数
    "BATCH_SIZE": 32,   # 推理批大小
    "CONF_THRESH": 0.001,   # 置信度阈值
    "OUTPUT_DIR": SCRIPT_DIR.parent / "caches",    # 输出目录
    "SUSPECTS_DIR": SCRIPT_DIR.parent / "caches/suspects", # 可疑图片输出目录
    "CLEAN_DIR": SCRIPT_DIR.parent / "caches/clean",   # 干净图片输出目录
    "SAVE_IMGS": True,  # 是否保存图片
    "CACHE_DIR": SCRIPT_DIR.parent / "caches/cache",  # 缓存目录
    "TRAIN_PARAMS": {  # 默认超参，可自行调
        "epochs": 150,
        "imgsz": 224,
        "batch": 32,
        "patience": 20,
        "device": "0"  # 多 GPU 写 "0,1,2"
    },

    # K-Means参数
    "KMEANS_CONFIG": {
        "n_clusters_ratio": 0.1,  # 推荐: 0.05-0.2
        "min_clusters": 2,  # 推荐: 2-5
        "max_clusters": 20,  # 推荐: 10-50
        "contamination": 0.1,  # 推荐: 0.05-0.2
        "random_state": 42  # 推荐: 固定值确保可重复
    },

    # Isolation Forest参数
    "ISOLATION_CONFIG": {
        "contamination": 0.1,  # 默认: "auto", 推荐: 0.05-0.2
        "n_estimators": 100,  # 默认: 100
        "random_state": 42,  # 默认: None
        "n_jobs": -1  # 默认: None
    },

    # 集成投票参数
    "ENSEMBLE_CONFIG": {
        "voting_threshold": 2,  # 推荐: 1(宽松) 2(平衡) 3(严格)
        "quality_score_weight": {  # 权重建议根据你的数据调整
            "cleanlab": 0.5,  # CleanLab通常最可靠
            "kmeans": 0.3,  # K-Means适合类内异常
            "isolation": 0.2  # Isolation Forest找全局异常
        }
    }

}


# ───────────────────────────────────────────────────────────────
# 🔄 数据加载和推理函数
# ───────────────────────────────────────────────────────────────
def load_dataset(data_dir: Path) -> Tuple[List[Path], List[int], Dict[int, str]]:
    """读取文件夹式数据集，支持递归搜索图片文件。"""
    print(f"📂 正在读取数据集: {data_dir}")
    if not data_dir.exists():
        raise ValueError(f"❌ 数据目录不存在: {data_dir}")

    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"❌ 未在 {data_dir} 下找到任何类别子文件夹")

    label_map = {idx: p.name for idx, p in enumerate(class_dirs)}
    img_paths, labels = [], []

    # 支持的图片格式（Windows系统使用不区分大小写的方式）
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    # 添加进度条显示类别处理过程
    for idx, cls_dir in enumerate(tqdm(class_dirs, desc="📁 扫描类别目录")):
        print(f"   📁 处理类别: {cls_dir.name}")

        # 使用集合避免重复文件
        paths_set = set()
        for ext in img_extensions:
            # 只搜索小写扩展名，在Windows上会同时匹配大小写
            for path in cls_dir.rglob(f"*{ext}"):
                paths_set.add(path)

        paths = list(paths_set)

        if paths:
            img_paths.extend(paths)
            labels.extend([idx] * len(paths))
            print(f"      找到 {len(paths)} 张图片")
        else:
            print(f"      ⚠️ 未找到任何图片文件")

    print(f"✅ 数据加载完成，共 {len(img_paths)} 张图片，{len(label_map)} 个类别")
    print(f"📊 类别分布:")
    for idx, name in label_map.items():
        count = sum(1 for label in labels if label == idx)
        print(f"   {idx}: {name} - {count} 张")

    return img_paths, labels, label_map


def _batch_predict(
        models: List[YOLO],
        paths: List[Path],
        batch_size: int,
        conf: float,
        fold_id: int = 0,
) -> np.ndarray:
    """集成平均多个模型的 Softmax 概率，返回 [N, C]
    注意：此函数需要使用YOLO分类模型，不是检测模型
    """
    pred_accum = None

    # 为每个模型添加进度条
    for model_idx, model in enumerate(models):
        probs_list = []

        # 计算总批次数
        total_batches = (len(paths) + batch_size - 1) // batch_size

        # 添加批次处理进度条
        batch_pbar = tqdm(
            range(0, len(paths), batch_size),
            desc=f"🔮 Fold {fold_id} - 模型 {model_idx + 1}/{len(models)} 推理",
            total=total_batches,
            leave=False
        )

        for i in batch_pbar:
            batch = [str(p) for p in paths[i: i + batch_size]]
            results = model.predict(batch, conf=conf, verbose=False, device=0)

            for r in results:
                # 检查是否为分类模型结果
                if hasattr(r, 'probs') and r.probs is not None:
                    # 分类模型
                    p = r.probs.data.cpu().numpy() if hasattr(r.probs, "data") else r.probs.cpu().numpy()
                    probs_list.append(p)
                elif hasattr(r, 'boxes') and r.boxes is not None:
                    # 检测模型 - 需要转换为分类概率
                    if model_idx == 0 and i == 0:  # 只在第一次警告
                        tqdm.write("⚠️ 检测到使用的是检测模型，尝试转换为分类概率...")

                    if len(r.boxes.cls) > 0:
                        # 取置信度最高的检测框的类别
                        max_conf_idx = r.boxes.conf.argmax()
                        cls_id = int(r.boxes.cls[max_conf_idx])
                        conf_score = float(r.boxes.conf[max_conf_idx])

                        # 创建one-hot概率向量
                        n_classes = len(model.names)
                        probs = np.zeros(n_classes)
                        probs[cls_id] = conf_score
                        # 将剩余概率平均分配给其他类别
                        remaining_prob = 1.0 - conf_score
                        other_prob = remaining_prob / (n_classes - 1)
                        for j in range(n_classes):
                            if j != cls_id:
                                probs[j] = other_prob
                        probs_list.append(probs)
                    else:
                        # 没有检测到任何对象，创建均匀分布
                        n_classes = len(model.names)
                        probs = np.ones(n_classes) / n_classes
                        probs_list.append(probs)
                else:
                    raise ValueError(f"❌ 无法从模型结果中提取概率信息。请确保使用的是YOLO分类模型。")

            # 更新进度条后缀信息
            batch_pbar.set_postfix({
                'processed': f"{min(i + batch_size, len(paths))}/{len(paths)}"
            })

        probs_arr = np.vstack(probs_list)
        pred_accum = probs_arr if pred_accum is None else pred_accum + probs_arr

    return pred_accum / len(models)

# 评估所需
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import matplotlib.pyplot as plt

# 说明：期望在调用模块中已定义全局 CONFIG
# 依赖外部函数 _batch_predict（与原脚本保持一致）

# ───────────────────────────────────────────────────────────────
# 1. 训练辅助函数
# ───────────────────────────────────────────────────────────────

def train_kfold_models(
    img_paths: List[Path],
    labels: List[int],
    label_map: Dict[int, str],
    n_folds: int = CONFIG["KFOLD"],
    params: dict = CONFIG["TRAIN_PARAMS"],
) -> List[Path]:
    """使用 **StratifiedKFold** 训练 ``n_folds`` 个 YOLO 分类模型。

    第 ``i`` 折模型不会看到第 ``i`` 折的验证样本，因此其 ``best.pt``
    随后可用于生成 Out‑of‑Fold (OOF) 预测。

    返回按折序（0 开始）的权重文件路径列表。
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    df = pd.DataFrame({"path": img_paths, "label": labels})

    base_ckpt = "yolov8n-cls.pt"  # 预训练权重，可按需求替换
    weight_paths: List[Path] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(img_paths, labels)):
        fold_dir = Path("runs/cls/kfold_train") / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # 保存训练 / 验证文件列表；使用 ``iloc`` 并转为 ``str`` 避免 dtype 问题
        (fold_dir / "train.txt").write_text("\n".join(df.iloc[tr_idx]["path"].astype(str)))
        (fold_dir / "val.txt").write_text("\n".join(df.iloc[val_idx]["path"].astype(str)))

        # YOLO 需要类别名称（字符串）
        names = [label_map[i] for i in sorted(label_map.keys())]
        data_yaml = {
            "train": str((fold_dir / "train.txt").resolve()),
            "val":   str((fold_dir / "val.txt").resolve()),
            "names": names,
        }
        (fold_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

        best_ckpt = fold_dir / "weights" / "best.pt"
        if not best_ckpt.exists():  # 若已训练则跳过
            model = YOLO(base_ckpt)
            model.train(
                data=str(fold_dir / "data.yaml"),
                epochs=params["epochs"],
                imgsz=params["imgsz"],
                batch=params["batch"],
                patience=params["patience"],
                project=str(fold_dir),
                name="",  # 留空 → 文件夹名已含 foldX
                exist_ok=True,
                device=params["device"],
            )
        weight_paths.append(best_ckpt)

    return weight_paths
# ───────────────────────────────────────────────────────────────
# 2. 推理辅助函数
# ───────────────────────────────────────────────────────────────

def kfold_predict(
    img_paths: List[Path],
    labels: List[int],
    weight_paths: List[Path],
    n_folds: int,
    batch_size: int,
    conf_thresh: float,
    use_cache: bool = CONFIG.get("USE_CACHE", True),
    dataset_name: str = CONFIG["DATA_DIR"].name,
) -> Tuple[np.ndarray, np.ndarray]:
    """使用预训练折模型生成 Out‑of‑Fold 概率。

    每个验证折使用 *除本折外* 的 ``n_folds-1`` 个模型组成集成进行预测，
    以避免信息泄漏。
    """
    # ------------------------- 缓存快速路径 -------------------------
    model_names = "_".join([w.parent.parent.parent.name for w in weight_paths])
    cache_dir = CONFIG["CACHE_DIR"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{n_folds}_{dataset_name}_{model_names}.npz"

    if use_cache and cache_path.exists():
        data = np.load(cache_path)
        return data["y_true"], data["y_pred"]

    # ------------------------ 载入所有权重 -------------------------
    models: List[YOLO] = []
    for w in weight_paths:
        if not w.exists():
            raise FileNotFoundError(f"未找到权重文件: {w}")
        models.append(YOLO(str(w)))

    n_classes = len(models[0].names)
    if not all(len(m.names) == n_classes for m in models[1:]):
        raise ValueError("所有折模型必须拥有相同的类别数量。")

    y_true = np.array(labels, dtype=int)
    y_pred = np.zeros((len(img_paths), n_classes), dtype=float)

    # ---------------------- 重复训练时的折划分 ----------------------
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_iter = tqdm(
        enumerate(skf.split(img_paths, labels)),
        total=n_folds,
        desc="K‑Fold 推理",
    )

    for fold, (_, val_idx) in fold_iter:
        val_paths = [img_paths[i] for i in val_idx]
        start = time.time()

        # 排除与当前验证折同源的模型
        ensemble = [m for i, m in enumerate(models) if i != fold]
        y_pred[val_idx] = _batch_predict(
            ensemble,
            val_paths,
            batch_size,
            conf_thresh,
            fold_id=fold,
        )

        fold_iter.set_postfix(samples=len(val_idx), t=f"{time.time()-start:.1f}s")

    np.savez(cache_path, y_true=y_true, y_pred=y_pred)
    return y_true, y_pred


# ───────────────────────────────────────────────────────────────
# 3. 评估辅助函数
# ───────────────────────────────────────────────────────────────

def evaluate_oof(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: Dict[int, str],
    save_dir: Optional[Path] = None,
    plot: bool = True,
) -> Dict[str, float]:
    """评估 OOF 预测结果并可视化。

    参数
    -----
    y_true : np.ndarray
        真实标签，一维整数数组。
    y_pred : np.ndarray
        预测概率，形状为 ``[N, C]``。
    label_map : Dict[int, str]
        ``{label_id: label_name}`` 映射，用于混淆矩阵坐标。
    save_dir : Path | None
        图表及报告保存目录，默认为 ``runs/cls/metrics``。
    plot : bool
        是否绘制并保存图表（混淆矩阵）。

    返回
    -----
    Dict[str, float]
        关键指标字典：accuracy、macro_f1、weighted_f1、kappa。
    """
    if save_dir is None:
        save_dir = Path("runs/cls/metrics")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 将概率转换为类别
    y_hat = y_pred.argmax(axis=1)

    acc = accuracy_score(y_true, y_hat)
    macro_f1 = f1_score(y_true, y_hat, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_hat, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_hat)

    # 分类报告保存为 txt，方便对齐检测
    report = classification_report(
        y_true,
        y_hat,
        target_names=[label_map[i] for i in sorted(label_map.keys())],
        digits=4,
    )
    (save_dir / "classification_report.txt").write_text(report)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_hat, labels=list(label_map.keys()))

    if plot:
        _plot_confusion_matrix(
            cm,
            class_names=[label_map[i] for i in sorted(label_map.keys())],
            title=f"Confusion Matrix | Acc={acc:.3f}",
            save_path=save_dir / "confusion_matrix.png",
        )

    # 多类别 ROC‑AUC：使用 macro‑average（OvR）
    try:
        roc_auc = roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")  # 若概率全零等情况无法计算

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "kappa": kappa,
        "roc_auc_macro": roc_auc,
    }

    # 保存指标到 csv（追加 / 更新）
    metrics_df = pd.DataFrame([metrics])
    csv_path = save_dir / "metrics.csv"
    if csv_path.exists():
        metrics_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        metrics_df.to_csv(csv_path, index=False)

    return metrics


# ───────────────────────────────────────────────────────────────
# 4. 私有工具函数
# ───────────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: Path,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """绘制并保存混淆矩阵。"""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # 在每个单元格上写数字
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)



# ───────────────────────────────────────────────────────────────
# 🆕 异常检测算法
# ───────────────────────────────────────────────────────────────
def run_enhanced_cleanlab(
        y_true: np.ndarray,
        pred_probs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    print("🧹 运行 CleanLab...")

    cl = CleanLearning(
        DummyClassifier(strategy="prior"),
        seed=42,
        cv_n_folds=1
    )

    issues = cl.find_label_issues(
        labels=y_true,
        pred_probs=pred_probs,
    )

    # 获取可疑样本索引和分数
    # 使用 CleanLab 返回的 sample_index 列, 仅保留被判定为问题标签的样本
    suspect_indices = issues.loc[issues["is_label_issue"], "sample_index"].to_numpy()
    # 你可以根据实际 DataFrame 列名选择分数字段
    quality_scores = issues["label_quality"].to_numpy() if "label_quality" in issues.columns else np.ones(len(y_true))

    print(f"🔍 CleanLab检测到 {len(suspect_indices)} 个可疑样本")
    return suspect_indices, quality_scores


def run_kmeans_detection(
        y_true: np.ndarray,
        pred_probs: np.ndarray,
        config: dict = CONFIG["KMEANS_CONFIG"]
) -> Tuple[np.ndarray, np.ndarray]:
    """基于K-Means的类内异常检测"""
    print("🎯 运行 K-Means 类内异常检测...")

    suspect_indices = []
    anomaly_scores = np.ones(len(y_true))  # 默认分数为1(正常)

    unique_labels = np.unique(y_true)

    for label in tqdm(unique_labels, desc="K-Means类内检测"):
        # 获取当前类别的样本
        class_mask = y_true == label
        class_indices = np.where(class_mask)[0]
        class_probs = pred_probs[class_mask]

        if len(class_indices) < config["min_clusters"]:
            continue

        # 动态决定聚类数
        n_samples = len(class_indices)
        n_clusters = max(
            config["min_clusters"],
            min(config["max_clusters"], int(n_samples * config["n_clusters_ratio"]))
        )

        # K-Means聚类
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=config["random_state"],
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(class_probs)

        # 计算每个样本到其聚类中心的距离
        distances = np.min(kmeans.transform(class_probs), axis=1)

        # 标记距离最大的样本为异常
        contamination_count = max(1, int(len(class_indices) * config["contamination"]))
        anomaly_threshold = np.percentile(distances, (1 - config["contamination"]) * 100)

        class_anomalies = class_indices[distances > anomaly_threshold]
        suspect_indices.extend(class_anomalies)

        # 记录异常分数 (距离越大分数越低)
        max_dist = distances.max()
        for i, dist in enumerate(distances):
            anomaly_scores[class_indices[i]] = 1 - (dist / max_dist) if max_dist > 0 else 1.0

    suspect_indices = np.array(suspect_indices)
    print(f"🎯 K-Means检测到 {len(suspect_indices)} 个类内异常样本")

    return suspect_indices, anomaly_scores


def run_isolation_forest(
        y_true: np.ndarray,
        pred_probs: np.ndarray,
        config: dict = CONFIG["ISOLATION_CONFIG"]
) -> Tuple[np.ndarray, np.ndarray]:
    """Isolation Forest全局异常检测"""
    print("🌲 运行 Isolation Forest 全局异常检测...")

    # 特征工程：结合预测概率和标签信息
    features = []

    # 1. 原始预测概率
    features.append(pred_probs)

    # 2. 预测置信度 (最大概率)
    confidence = pred_probs.max(axis=1).reshape(-1, 1)
    features.append(confidence)

    # 3. 预测熵 (不确定性)
    entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-8), axis=1).reshape(-1, 1)
    features.append(entropy)

    # 4. 标签一致性 (真实标签的预测概率)
    label_consistency = pred_probs[np.arange(len(y_true)), y_true].reshape(-1, 1)
    features.append(label_consistency)

    # 合并特征
    X = np.hstack(features)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 降维 (可选)
    if X_scaled.shape[1] > 10:
        pca = PCA(n_components=min(10, X_scaled.shape[1]))
        X_scaled = pca.fit_transform(X_scaled)

    # Isolation Forest
    iso_forest = IsolationForest(
        contamination=config["contamination"],
        n_estimators=config["n_estimators"],
        random_state=config["random_state"],
        n_jobs=config["n_jobs"]
    )

    anomaly_labels = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.score_samples(X_scaled)

    # 转换分数到 [0, 1] 范围 (分数越低越异常)
    anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())

    suspect_indices = np.where(anomaly_labels == -1)[0]

    print(f"🌲 Isolation Forest检测到 {len(suspect_indices)} 个全局异常样本")
    return suspect_indices, anomaly_scores


# ───────────────────────────────────────────────────────────────
# 🆕 集成决策
# ───────────────────────────────────────────────────────────────
def ensemble_decision(
        y_true: np.ndarray,
        cleanlab_suspects: np.ndarray,
        cleanlab_scores: np.ndarray,
        kmeans_suspects: np.ndarray,
        kmeans_scores: np.ndarray,
        isolation_suspects: np.ndarray,
        isolation_scores: np.ndarray,
        config: dict = CONFIG["ENSEMBLE_CONFIG"]
) -> pd.DataFrame:
    """集成多个算法的异常检测结果"""
    print("🤝 集成多算法检测结果...")

    n_samples = len(y_true)
    results = []

    # 创建投票矩阵
    votes = np.zeros((n_samples, 3))  # [cleanlab, kmeans, isolation]
    votes[cleanlab_suspects, 0] = 1
    votes[kmeans_suspects, 1] = 1
    votes[isolation_suspects, 2] = 1

    weights = np.array([
        config["quality_score_weight"]["cleanlab"],
        config["quality_score_weight"]["kmeans"],
        config["quality_score_weight"]["isolation"]
    ])

    for i in range(n_samples):
        # 投票统计
        vote_count = votes[i].sum()
        weighted_vote = np.dot(votes[i], weights)

        # 综合质量分数 (越低越可疑)
        composite_score = (
                cleanlab_scores[i] * config["quality_score_weight"]["cleanlab"] +
                kmeans_scores[i] * config["quality_score_weight"]["kmeans"] +
                isolation_scores[i] * config["quality_score_weight"]["isolation"]
        )

        # 最终决策
        is_suspect = vote_count >= config["voting_threshold"]

        results.append({
            "index": i,
            "cleanlab_suspect": i in cleanlab_suspects,
            "kmeans_suspect": i in kmeans_suspects,
            "isolation_suspect": i in isolation_suspects,
            "vote_count": int(vote_count),
            "weighted_vote": weighted_vote,
            "composite_score": composite_score,
            "is_suspect": is_suspect
        })

    df_results = pd.DataFrame(results)

    # 统计信息
    total_suspects = df_results["is_suspect"].sum()
    print(f"🤝 集成结果: {total_suspects}/{n_samples} 样本被标记为可疑 ({total_suspects / n_samples * 100:.2f}%)")

    # 各算法贡献统计
    print("📊 各算法检测统计:")
    print(f"   CleanLab: {len(cleanlab_suspects)} 个")
    print(f"   K-Means: {len(kmeans_suspects)} 个")
    print(f"   Isolation Forest: {len(isolation_suspects)} 个")
    print(f"   最终集成: {total_suspects} 个")

    return df_results.sort_values("composite_score")


# ───────────────────────────────────────────────────────────────
# 🔄 导出函数
# ───────────────────────────────────────────────────────────────
def export_enhanced_results(
        df_ensemble: pd.DataFrame,
        img_paths: List[Path],
        label_map: Dict[int, str],
        y_true: np.ndarray,
        output_dir: Path = CONFIG["OUTPUT_DIR"],
        suspects_dir: Path = CONFIG["SUSPECTS_DIR"],
        clean_dir: Path = CONFIG["CLEAN_DIR"],
        save_suspects: bool = CONFIG["SAVE_IMGS"],
) -> None:
    """导出结果：保存CSV和复制图片"""
    print("💾 开始导出增强版结果...")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_suspects:
        suspects_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    # 准备完整的结果DataFrame
    suspect_indices = set(df_ensemble[df_ensemble["is_suspect"]]["index"].tolist())

    print("📝 准备增强版结果数据...")
    results = []
    for idx, img_path in enumerate(tqdm(img_paths, desc="处理结果数据")):
        row_data = df_ensemble[df_ensemble["index"] == idx].iloc[0]
        is_suspect = row_data["is_suspect"]

        results.append({
            "index": idx,
            "image_path": str(img_path),
            "true_label": y_true[idx],
            "true_label_name": label_map[y_true[idx]],
            "is_suspect": is_suspect,
            "cleanlab_suspect": row_data["cleanlab_suspect"],
            "kmeans_suspect": row_data["kmeans_suspect"],
            "isolation_suspect": row_data["isolation_suspect"],
            "vote_count": row_data["vote_count"],
            "weighted_vote": row_data["weighted_vote"],
            "composite_score": row_data["composite_score"]
        })

    df_results = pd.DataFrame(results)

    # 保存CSV
    csv_path = output_dir / "enhanced_label_issues.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"💾 增强版结果已保存到: {csv_path}")

    # 复制可疑图片
    if save_suspects and len(suspect_indices) > 0:
        print("📋 复制可疑图片...")
        for idx in tqdm(suspect_indices, desc="复制可疑图片"):
            src_path = img_paths[idx]
            true_label_name = label_map[y_true[idx]]

            # 添加算法标识
            row_data = df_ensemble[df_ensemble["index"] == idx].iloc[0]
            algo_flags = []
            if row_data["cleanlab_suspect"]: algo_flags.append("CL")
            if row_data["kmeans_suspect"]: algo_flags.append("KM")
            if row_data["isolation_suspect"]: algo_flags.append("IF")
            algo_str = "-".join(algo_flags)

            dst_path = suspects_dir / f"{idx}_{true_label_name}_{algo_str}_{src_path.name}"
            shutil.copy2(src_path, dst_path)
        print(f"✅ 可疑图片已复制到: {suspects_dir}")

    # 复制干净图片
    clean_indices = [i for i in range(len(img_paths)) if i not in suspect_indices]
    print(f"📋 复制干净图片 ({len(clean_indices)} 张)...")

    # 按类别组织干净图片，添加进度条
    for idx in tqdm(clean_indices, desc="复制干净图片"):
        src_path = img_paths[idx]
        true_label = y_true[idx]
        label_name = label_map[true_label]

        # 创建类别目录
        class_dir = clean_dir / label_name
        class_dir.mkdir(parents=True, exist_ok=True)

        dst_path = class_dir / src_path.name
        shutil.copy2(src_path, dst_path)

    print(f"✅ 干净图片已复制到: {clean_dir}")
    print(f"📊 最终统计:")
    print(f"   总图片数: {len(img_paths)}")
    print(f"   可疑图片: {len(suspect_indices)} ({len(suspect_indices) / len(img_paths) * 100:.2f}%)")
    print(f"   干净图片: {len(clean_indices)} ({len(clean_indices) / len(img_paths) * 100:.2f}%)")

    # 按类别显示可疑图片分布
    if len(suspect_indices) > 0:
        print(f"📊 各类别可疑图片分布:")
        suspect_by_class = {}
        for idx in suspect_indices:
            true_label = y_true[idx]
            label_name = label_map[true_label]
            suspect_by_class[label_name] = suspect_by_class.get(label_name, 0) + 1

        for label_name, count in sorted(suspect_by_class.items()):
            total_in_class = sum(1 for label in y_true if label_map[label] == label_name)
            print(f"   {label_name}: {count}/{total_in_class} ({count / total_in_class * 100:.1f}%)")


# ───────────────────────────────────────────────────────────────
# 主流程
# ───────────────────────────────────────────────────────────────
def main():
    """增强版主流程"""
    start_time = time.time()

    print("🎯 开始增强版AffectNet标签清洗")
    print("=" * 70)

    try:
        # 1. 加载数据
        print("\n📊 步骤 1/7: 加载数据集")
        img_paths, labels, label_map = load_dataset(CONFIG["DATA_DIR"])

        # 2. 模型推理
        print("\n🚀 步骤 2/7: K折交叉验证推理")

        weight_paths = train_kfold_models(img_paths, labels)

        y_true, pred_probs = kfold_predict(
            img_paths, labels, weight_paths,
            CONFIG["KFOLD"], CONFIG["BATCH_SIZE"], CONFIG["CONF_THRESH"]
        )

        # 3. CleanLab检测
        print("\n🧹 步骤 3/7: CleanLab异常检测")
        cleanlab_suspects, cleanlab_scores = run_enhanced_cleanlab(y_true, pred_probs)

        # 4. K-Means检测
        print("\n🎯 步骤 4/7: K-Means类内异常检测")
        kmeans_suspects, kmeans_scores = run_kmeans_detection(y_true, pred_probs)

        # 5. Isolation Forest检测
        print("\n🌲 步骤 5/7: Isolation Forest全局检测")
        isolation_suspects, isolation_scores = run_isolation_forest(y_true, pred_probs)

        # 6. 集成决策
        print("\n🤝 步骤 6/7: 集成多算法结果")
        df_ensemble = ensemble_decision(
            y_true, cleanlab_suspects, cleanlab_scores,
            kmeans_suspects, kmeans_scores,
            isolation_suspects, isolation_scores
        )

        # 7. 导出结果
        print("\n💾 步骤 7/7: 导出增强版结果")
        output_dir = CONFIG["OUTPUT_DIR"]
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存详细结果
        csv_path = output_dir / "enhanced_label_issues.csv"

        # 添加图片路径和标签信息
        df_final = df_ensemble.copy()
        df_final["image_path"] = [str(p) for p in img_paths]
        df_final["true_label"] = y_true
        df_final["true_label_name"] = [label_map[l] for l in y_true]
        df_final["pred_label"] = pred_probs.argmax(axis=1)
        df_final["pred_label_name"] = [label_map[l] for l in pred_probs.argmax(axis=1)]

        df_final.to_csv(csv_path, index=False)
        print(f"💾 增强版结果已保存: {csv_path}")

        # 复制干净图片 (基于集成结果)
        clean_dir = CONFIG["CLEAN_DIR"]
        clean_dir.mkdir(parents=True, exist_ok=True)

        clean_samples = df_final[~df_final["is_suspect"]]
        print(f"📋 复制 {len(clean_samples)} 张干净图片...")

        for _, row in tqdm(clean_samples.iterrows(), total=len(clean_samples), desc="复制干净图片"):
            src_path = Path(row["image_path"])
            label_name = row["true_label_name"]
            class_dir = clean_dir / label_name
            class_dir.mkdir(exist_ok=True)
            dst_path = class_dir / src_path.name
            shutil.copy2(src_path, dst_path)

        total_time = time.time() - start_time
        print(f"\n🎉 增强版清洗完成！总耗时: {total_time:.1f}秒")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        raise


if __name__ == "__main__":
    main()