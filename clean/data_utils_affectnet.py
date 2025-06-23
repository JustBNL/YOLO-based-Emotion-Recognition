from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil


def load_dataset(data_dir: Path) -> Tuple[List[Path], List[int], Dict[int, str]]:
    print(f"📂 正在读取数据集: {data_dir}")
    if not data_dir.exists():
        raise ValueError(f"❌ 数据目录不存在: {data_dir}")

    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"❌ 未在 {data_dir} 下找到任何类别子文件夹")

    label_map = {idx: p.name for idx, p in enumerate(class_dirs)}
    img_paths, labels = [], []

    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    for idx, cls_dir in enumerate(tqdm(class_dirs, desc="📁 扫描类别目录")):
        print(f"   📁 处理类别: {cls_dir.name}")
        paths_set = set()
        for ext in img_extensions:
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
    print("📊 类别分布:")
    for idx, name in label_map.items():
        count = sum(1 for label in labels if label == idx)
        print(f"   {idx}: {name} - {count} 张")

    return img_paths, labels, label_map


def export_enhanced_results(
    df_ensemble: pd.DataFrame,
    img_paths: List[Path],
    label_map: Dict[int, str],
    y_true: np.ndarray,
    output_dir: Path,
    suspects_dir: Path,
    clean_dir: Path,
    save_suspects: bool = True
) -> None:
    print("💾 开始导出增强版结果...")
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_suspects:
        suspects_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

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
    csv_path = output_dir / "enhanced_label_issues.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"💾 增强版结果已保存到: {csv_path}")

    if save_suspects and len(suspect_indices) > 0:
        print("📋 复制可疑图片...")
        for idx in tqdm(suspect_indices, desc="复制可疑图片"):
            src_path = img_paths[idx]
            true_label_name = label_map[y_true[idx]]
            row_data = df_ensemble[df_ensemble["index"] == idx].iloc[0]
            algo_flags = []
            if row_data["cleanlab_suspect"]: algo_flags.append("CL")
            if row_data["kmeans_suspect"]: algo_flags.append("KM")
            if row_data["isolation_suspect"]: algo_flags.append("IF")
            algo_str = "-".join(algo_flags)
            dst_path = suspects_dir / f"{idx}_{true_label_name}_{algo_str}_{src_path.name}"
            shutil.copy2(src_path, dst_path)
        print(f"✅ 可疑图片已复制到: {suspects_dir}")

    clean_indices = [i for i in range(len(img_paths)) if i not in suspect_indices]
    print(f"📋 复制干净图片 ({len(clean_indices)} 张)...")
    for idx in tqdm(clean_indices, desc="复制干净图片"):
        src_path = img_paths[idx]
        true_label = y_true[idx]
        label_name = label_map[true_label]
        class_dir = clean_dir / label_name
        class_dir.mkdir(parents=True, exist_ok=True)
        dst_path = class_dir / src_path.name
        shutil.copy2(src_path, dst_path)

    print(f"✅ 干净图片已复制到: {clean_dir}")
    print(f"📊 最终统计:")
    print(f"   总图片数: {len(img_paths)}")
    print(f"   可疑图片: {len(suspect_indices)} ({len(suspect_indices) / len(img_paths) * 100:.2f}%)")
    print(f"   干净图片: {len(clean_indices)} ({len(clean_indices) / len(img_paths) * 100:.2f}%)")

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
