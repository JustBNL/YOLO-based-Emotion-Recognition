from pathlib import Path
import time
from kfold_train_utils import train_kfold_models, kfold_predict  # 确保导入了新的训练和预测函数
from data_utils_affectnet import load_dataset, export_enhanced_results
from anomaly_detection import run_enhanced_cleanlab, run_kmeans_detection, run_isolation_forest, ensemble_decision

# ───────────────────────────────────────────────────────────────
# 配置参数 (保留原始脚本中的CONFIG)
# ───────────────────────────────────────────────────────────────
SCRIPT_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = SCRIPT_DIR.parent

CONFIG: dict = {
    "DATA_DIR": PROJECT_ROOT / "datasets/cls/processed/affectnet-1",  # 数据集根目录
    "KFOLD": 5,  # K折交叉验证折数
    "BATCH_SIZE": 32,  # 推理批大小
    "CONF_THRESH": 0.001,  # 置信度阈值
    "OUTPUT_DIR": SCRIPT_DIR / "caches",  # 输出目录
    "SUSPECTS_DIR": SCRIPT_DIR / "caches/suspects",  # 可疑图片输出目录
    "CLEAN_DIR": SCRIPT_DIR / "caches/clean",  # 干净图片输出目录
    "SAVE_IMGS": True,  # 是否保存图片
    "CACHE_DIR": SCRIPT_DIR / "caches/cache",  # 缓存目录

    # 缓存控制
    "USE_CACHE": True,  # 是否使用缓存功能

    "TRAIN_PARAMS": {
        "epochs": 150,
        "imgsz": 224,
        "batch": 32,
        "patience": 20,
        "device": "0"
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
        "voting_threshold": 2,  # 至少2个算法同意才标记为可疑(1-3)
        "score_threshold": 0.4,  # 综合分数低于0.4也标记为可疑

        # 算法权重
        "quality_score_weight": {
            "cleanlab": 0.5,  # CleanLab通常最可靠
            "kmeans": 0.3,  # K-Means适合类内异常
            "isolation": 0.2  # Isolation Forest找全局异常
        }
    }
}


def main():
    start_time = time.time()
    print("🎯 开始增强版AffectNet标签清洗")
    print("=" * 70)

    # 输出缓存状态
    if CONFIG["USE_CACHE"]:
        print(f"📋 缓存功能: 启用 (缓存目录: {CONFIG['CACHE_DIR']})")
    else:
        print("📋 缓存功能: 禁用")

    try:
        # 1. 加载数据集
        print("\n📊 步骤 1/8: 加载数据集")
        img_paths, labels, label_map = load_dataset(CONFIG["DATA_DIR"])
        print(f"   加载完成: {len(img_paths)} 张图片, {len(label_map)} 个类别")

        # 2. K折交叉验证训练
        print("\n🚀 步骤 2/8: K折交叉验证训练")
        weight_paths = train_kfold_models(
            img_paths,
            labels,
            label_map,
            CONFIG["KFOLD"],
            CONFIG["TRAIN_PARAMS"],
            data_dir=CONFIG["DATA_DIR"],
            use_cache=CONFIG["USE_CACHE"]
        )
        print(f"   训练完成: {len(weight_paths)} 个模型权重")

        # 3. K折交叉验证推理
        print("\n🔍 步骤 3/8: K折交叉验证推理")
        y_true, pred_probs = kfold_predict(
            img_paths,
            labels,
            weight_paths,
            CONFIG["KFOLD"],
            CONFIG["BATCH_SIZE"],
            CONFIG["CONF_THRESH"],
            CONFIG,
            data_dir=CONFIG["DATA_DIR"],
            use_cache=CONFIG["USE_CACHE"]
        )
        print(f"   推理完成: 预测形状 {pred_probs.shape}")

        # 4. CleanLab异常检测
        print("\n🧹 步骤 4/8: CleanLab异常检测")
        cleanlab_suspects, cleanlab_scores = run_enhanced_cleanlab(y_true, pred_probs)
        print(f"   检测完成: {len(cleanlab_suspects)} 个可疑样本")

        # 5. K-Means类内异常检测
        print("\n🎯 步骤 5/8: K-Means类内异常检测")
        kmeans_suspects, kmeans_scores = run_kmeans_detection(y_true, pred_probs, CONFIG["KMEANS_CONFIG"])
        print(f"   检测完成: {len(kmeans_suspects)} 个可疑样本")

        # 6. Isolation Forest全局检测
        print("\n🌲 步骤 6/8: Isolation Forest全局检测")
        isolation_suspects, isolation_scores = run_isolation_forest(y_true, pred_probs, CONFIG["ISOLATION_CONFIG"])
        print(f"   检测完成: {len(isolation_suspects)} 个可疑样本")

        # 7. 集成多算法结果
        print("\n🤝 步骤 7/8: 集成多算法结果")
        df_ensemble = ensemble_decision(
            y_true, cleanlab_suspects, cleanlab_scores,
            kmeans_suspects, kmeans_scores,
            isolation_suspects, isolation_scores, CONFIG["ENSEMBLE_CONFIG"]
        )
        print(f"   集成完成: {len(df_ensemble)} 个样本处理")

        # 8. 导出增强版结果
        print("\n💾 步骤 8/8: 导出增强版结果")
        export_enhanced_results(
            df_ensemble,
            img_paths,
            label_map,
            y_true,
            CONFIG["OUTPUT_DIR"],
            CONFIG["SUSPECTS_DIR"],
            CONFIG["CLEAN_DIR"],
            CONFIG["SAVE_IMGS"]
        )

        # 统计信息
        suspects_count = len(df_ensemble[df_ensemble['is_suspect'] == True])
        clean_count = len(df_ensemble[df_ensemble['is_suspect'] == False])

        print(f"\n📈 清洗统计:")
        print(f"   总样本数: {len(df_ensemble)}")
        print(f"   可疑样本: {suspects_count} ({suspects_count / len(df_ensemble) * 100:.1f}%)")
        print(f"   干净样本: {clean_count} ({clean_count / len(df_ensemble) * 100:.1f}%)")

        total_time = time.time() - start_time
        print(f"\n🎉 增强版清洗完成！总耗时: {total_time:.1f}秒")
        print("=" * 70)

    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断执行")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()