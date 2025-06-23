from typing import Tuple
import warnings

warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from cleanlab.classification import CleanLearning
from sklearn.dummy import DummyClassifier
import numpy as np
import pandas as pd
from tqdm import tqdm


# CleanLab 异常检测
def run_enhanced_cleanlab(y_true: np.ndarray, pred_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """使用CleanLab进行标签异常检测"""
    print("🧹 运行 CleanLab...")

    try:
        # 确保输入数据的有效性
        if len(y_true) != len(pred_probs):
            raise ValueError(f"标签数量 ({len(y_true)}) 与预测概率数量 ({len(pred_probs)}) 不匹配")

        if pred_probs.shape[1] != len(np.unique(y_true)):
            print(f"⚠️ 警告: 预测概率维度 ({pred_probs.shape[1]}) 与类别数量 ({len(np.unique(y_true))}) 不匹配")

        # 创建CleanLearning实例
        cl = CleanLearning(
            DummyClassifier(strategy="prior"),
            seed=42,
            cv_n_folds=1
        )

        # 查找标签问题
        issues = cl.find_label_issues(
            labels=y_true,
            pred_probs=pred_probs,
        )

        # 获取可疑样本索引和分数
        suspect_indices = issues.query("is_label_issue == True").index.to_numpy()

        # 获取质量分数，如果不存在则使用默认值
        if "label_quality" in issues.columns:
            quality_scores = issues["label_quality"].to_numpy()
        else:
            # 使用预测概率的置信度作为质量分数
            quality_scores = pred_probs.max(axis=1)

        print(f"🔍 CleanLab检测到 {len(suspect_indices)} 个可疑样本")
        return suspect_indices, quality_scores

    except Exception as e:
        print(f"❌ CleanLab检测失败: {e}")
        # 返回空结果
        return np.array([]), np.ones(len(y_true))


# K-Means 类内异常检测
def run_kmeans_detection(y_true: np.ndarray, pred_probs: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """使用K-Means进行类内异常检测"""
    print("🎯 运行 K-Means 类内异常检测...")

    try:
        suspect_indices = []
        anomaly_scores = np.ones(len(y_true))  # 分数,默认分数为1(正常)越小越可疑
        unique_labels = np.unique(y_true)

        for label in tqdm(unique_labels, desc="K-Means类内检测", leave=False):
            class_mask = y_true == label
            class_indices = np.where(class_mask)[0]
            class_probs = pred_probs[class_mask]

            # 检查样本数量是否足够进行聚类
            if len(class_indices) < config["min_clusters"]:
                print(f"   跳过类别 {label}: 样本数量不足 ({len(class_indices)} < {config['min_clusters']})")
                continue

            n_samples = len(class_indices)
            n_clusters = max(
                config["min_clusters"],
                min(config["max_clusters"], int(n_samples * config["n_clusters_ratio"]))
            )

            # 执行K-Means聚类
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=config["random_state"],
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(class_probs)

            # 计算到最近聚类中心的距离
            distances = np.min(kmeans.transform(class_probs), axis=1)

            # 确定异常阈值
            contamination_ratio = config["contamination"]
            anomaly_threshold = np.percentile(distances, (1 - contamination_ratio) * 100)

            # 识别异常样本
            anomaly_mask = distances > anomaly_threshold
            class_anomalies = class_indices[anomaly_mask]
            suspect_indices.extend(class_anomalies)

            # 计算异常分数（距离越大，分数越低）
            max_dist = distances.max()
            if max_dist > 0:
                for i, dist in enumerate(distances):
                    anomaly_scores[class_indices[i]] = 1 - (dist / max_dist)

            print(f"   类别 {label}: {len(class_anomalies)}/{len(class_indices)} 个异常样本")

        suspect_indices = np.array(suspect_indices)
        print(f"🎯 K-Means检测到 {len(suspect_indices)} 个类内异常样本")

        return suspect_indices, anomaly_scores

    except Exception as e:
        print(f"❌ K-Means检测失败: {e}")
        return np.array([]), np.ones(len(y_true))


# Isolation Forest 全局异常检测
def run_isolation_forest(y_true: np.ndarray, pred_probs: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """使用Isolation Forest进行全局异常检测"""
    print("🌲 运行 Isolation Forest 全局异常检测...")

    try:
        # 构建特征向量
        features = []

        # 1. 原始预测概率
        features.append(pred_probs)

        # 2. 最大概率（置信度）
        confidence = pred_probs.max(axis=1).reshape(-1, 1)
        features.append(confidence)

        # 3. 熵（不确定性越高越不正确）
        entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-8), axis=1).reshape(-1, 1)
        features.append(entropy)

        # 4. 标签一致性（真实标签对应的预测概率）
        label_consistency = pred_probs[np.arange(len(y_true)), y_true].reshape(-1, 1)
        features.append(label_consistency)

        # 5. 预测标签与真实标签的差异
        pred_labels = pred_probs.argmax(axis=1)
        label_mismatch = (pred_labels != y_true).astype(float).reshape(-1, 1)
        features.append(label_mismatch)

        # 合并所有特征
        X = np.hstack(features)

        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 如果特征维度过高，使用PCA降维
        if X_scaled.shape[1] > 15:
            pca = PCA(n_components=min(15, X_scaled.shape[1]))
            X_scaled = pca.fit_transform(X_scaled)
            print(f"   使用PCA降维: {X.shape[1]} -> {X_scaled.shape[1]}")

        # 创建Isolation Forest模型
        iso_forest = IsolationForest(
            contamination=config["contamination"],
            n_estimators=config["n_estimators"],
            random_state=config["random_state"],
            n_jobs=config["n_jobs"]
        )

        # 训练并预测
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.score_samples(X_scaled)

        # 将异常分数标准化到[0,1]范围，分数越高越正常
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())

        # 获取异常样本索引
        suspect_indices = np.where(anomaly_labels == -1)[0]
        print(f"🌲 Isolation Forest检测到 {len(suspect_indices)} 个全局异常样本")

        return suspect_indices, anomaly_scores

    except Exception as e:
        print(f"❌ Isolation Forest检测失败: {e}")
        return np.array([]), np.ones(len(y_true))


# 集成多个算法的异常检测结果
def ensemble_decision(
        y_true: np.ndarray,
        cleanlab_suspects: np.ndarray,
        cleanlab_scores: np.ndarray,
        kmeans_suspects: np.ndarray,
        kmeans_scores: np.ndarray,
        isolation_suspects: np.ndarray,
        isolation_scores: np.ndarray,
        config: dict
) -> pd.DataFrame:
    print("🤝 集成多算法检测结果...")

    try:
        n_samples = len(y_true)
        results = []

        # 创建投票矩阵
        votes = np.zeros((n_samples, 3), dtype=int)
        votes[cleanlab_suspects, 0] = 1
        votes[kmeans_suspects, 1] = 1
        votes[isolation_suspects, 2] = 1

        # 算法权重
        weights = np.array([
            config["quality_score_weight"]["cleanlab"],
            config["quality_score_weight"]["kmeans"],
            config["quality_score_weight"]["isolation"]
        ])

        # 对每个样本进行集成决策
        for i in range(n_samples):
            vote_count = votes[i].sum()
            weighted_vote = np.dot(votes[i], weights)

            # 计算综合质量分数
            composite_score = (
                    cleanlab_scores[i] * config["quality_score_weight"]["cleanlab"] +
                    kmeans_scores[i] * config["quality_score_weight"]["kmeans"] +
                    isolation_scores[i] * config["quality_score_weight"]["isolation"]
            )

            # 简化的判断逻辑：投票 OR 低分数
            is_suspect = (
                    vote_count >= config["voting_threshold"] or  # 投票阈值
                    composite_score <= config["score_threshold"]  # 分数阈值
            )

            results.append({
                "index": i,
                "true_label": y_true[i],
                "cleanlab_suspect": i in cleanlab_suspects,
                "kmeans_suspect": i in kmeans_suspects,
                "isolation_suspect": i in isolation_suspects,
                "vote_count": int(vote_count),
                "weighted_vote": weighted_vote,
                "cleanlab_score": cleanlab_scores[i],
                "kmeans_score": kmeans_scores[i],
                "isolation_score": isolation_scores[i],
                "composite_score": composite_score,
                "is_suspect": is_suspect
            })

        # 创建结果DataFrame
        df_results = pd.DataFrame(results)
        total_suspects = df_results["is_suspect"].sum()

        print(f"🤝 集成结果: {total_suspects}/{n_samples} 样本被标记为可疑 ({total_suspects / n_samples * 100:.1f}%)")

        print("📊 各算法检测统计:")
        print(f"   CleanLab: {len(cleanlab_suspects)} 个 ({len(cleanlab_suspects) / n_samples * 100:.1f}%)")
        print(f"   K-Means: {len(kmeans_suspects)} 个 ({len(kmeans_suspects) / n_samples * 100:.1f}%)")
        print(f"   Isolation Forest: {len(isolation_suspects)} 个 ({len(isolation_suspects) / n_samples * 100:.1f}%)")
        print(f"   最终集成: {total_suspects} 个 ({total_suspects / n_samples * 100:.1f}%)")

        # 分析触发原因
        vote_triggered = df_results[df_results["vote_count"] >= config["voting_threshold"]].shape[0]
        score_triggered = df_results[df_results["composite_score"] <= config["score_threshold"]].shape[0]
        both_triggered = df_results[
            (df_results["vote_count"] >= config["voting_threshold"]) &
            (df_results["composite_score"] <= config["score_threshold"])
            ].shape[0]

        print(f"\n🎯 触发原因分析:")
        print(f"   投票触发: {vote_triggered} 个")
        print(f"   分数触发: {score_triggered} 个")
        print(f"   双重触发: {both_triggered} 个")

        # 按综合分数排序，分数越低越可疑
        return df_results.sort_values("composite_score", ascending=True)

    except Exception as e:
        print(f"❌ 集成决策失败: {e}")
        # 返回基本的DataFrame
        return pd.DataFrame({
            "index": range(len(y_true)),
            "true_label": y_true,
            "is_suspect": [False] * len(y_true),
            "composite_score": [1.0] * len(y_true)
        })