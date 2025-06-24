from pathlib import Path
import time
import sys
import logging
from datetime import datetime
from kfold_train_utils import train_kfold_models, kfold_predict
from data_utils_affectnet import load_dataset, export_enhanced_results
from anomaly_detection import run_enhanced_cleanlab, run_kmeans_detection, run_isolation_forest, ensemble_decision
from anomaly_visualization import AnomalyDetectionVisualizer


# ───────────────────────────────────────────────────────────────
# 日志配置类
# ───────────────────────────────────────────────────────────────
class DualOutput:
    """同时输出到控制台和文件的类"""

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # 确保立即写入文件

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def setup_logging(log_dir: Path) -> tuple:
    """设置日志系统"""
    log_dir.mkdir(parents=True, exist_ok=True)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"affectnet_cleaning_{timestamp}.log"

    # 创建双输出对象
    dual_output = DualOutput(log_file)

    # 重定向stdout
    original_stdout = sys.stdout
    sys.stdout = dual_output

    print(f"📝 日志系统已启动")
    print(f"📁 日志文件: {log_file}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return dual_output, original_stdout, log_file


def cleanup_logging(dual_output, original_stdout, log_file: Path):
    """清理日志系统"""
    print("=" * 70)
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 日志已保存至: {log_file}")

    # 恢复原始stdout
    sys.stdout = original_stdout
    dual_output.close()

    print(f"✅ 日志系统已关闭，日志文件: {log_file}")


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
    "LOG_DIR": SCRIPT_DIR / "logs",  # 日志目录
    "VISUALIZATION_DIR": SCRIPT_DIR / "visualizations",  # 可视化输出目录
    "SAVE_IMGS": True,  # 是否保存图片
    "CACHE_DIR": SCRIPT_DIR / "caches/cache",  # 缓存目录
    "ENABLE_VISUALIZATION": True,  # 是否启用可视化

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
        "voting_threshold": 2,  # 至少3个算法同意才标记为可疑(1-3)
        "score_threshold": 0.3,  # 综合分数低于0.1也标记为可疑

        # 算法权重
        "quality_score_weight": {
            "cleanlab": 0.5,  # CleanLab通常最可靠
            "kmeans": 0.1,  # K-Means适合类内异常,不适合计算分数
            "isolation": 0.4  # Isolation Forest找全局异常
        }
    }
}


def print_config_summary():
    """打印配置摘要"""
    print("⚙️ 配置摘要:")
    print(f"   数据目录: {CONFIG['DATA_DIR']}")
    print(f"   K折数: {CONFIG['KFOLD']}")
    print(f"   批大小: {CONFIG['BATCH_SIZE']}")
    print(f"   缓存功能: {'启用' if CONFIG['USE_CACHE'] else '禁用'}")
    print(f"   保存图片: {'是' if CONFIG['SAVE_IMGS'] else '否'}")
    print(f"   可视化: {'启用' if CONFIG['ENABLE_VISUALIZATION'] else '禁用'}")
    print(f"   可视化目录: {CONFIG['VISUALIZATION_DIR']}")
    print(f"   训练轮数: {CONFIG['TRAIN_PARAMS']['epochs']}")
    print(f"   图片尺寸: {CONFIG['TRAIN_PARAMS']['imgsz']}")
    print(f"   设备: {CONFIG['TRAIN_PARAMS']['device']}")
    print("-" * 50)


def main():
    # 设置日志系统
    dual_output, original_stdout, log_file = setup_logging(CONFIG["LOG_DIR"])

    start_time = time.time()

    try:
        print("🎯 开始增强版AffectNet标签清洗")
        print_config_summary()

        # 1. 加载数据集
        print("\n📊 步骤 1/9: 加载数据集")
        img_paths, labels, label_map = load_dataset(CONFIG["DATA_DIR"])
        print(f"   加载完成: {len(img_paths)} 张图片, {len(label_map)} 个类别")

        # 2. K折交叉验证训练
        print("\n🚀 步骤 2/9: K折交叉验证训练")
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
        print("\n🔍 步骤 3/9: K折交叉验证推理")
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
        print("\n🧹 步骤 4/9: CleanLab异常检测")
        cleanlab_suspects, cleanlab_scores = run_enhanced_cleanlab(y_true, pred_probs)
        print(f"   检测完成: {len(cleanlab_suspects)} 个可疑样本")

        # 5. K-Means类内异常检测
        print("\n🎯 步骤 5/9: K-Means类内异常检测")
        kmeans_suspects, kmeans_scores = run_kmeans_detection(y_true, pred_probs, CONFIG["KMEANS_CONFIG"])
        print(f"   检测完成: {len(kmeans_suspects)} 个可疑样本")

        # 6. Isolation Forest全局检测
        print("\n🌲 步骤 6/9: Isolation Forest全局检测")
        isolation_suspects, isolation_scores = run_isolation_forest(y_true, pred_probs, CONFIG["ISOLATION_CONFIG"])
        print(f"   检测完成: {len(isolation_suspects)} 个可疑样本")

        # 7. 集成多算法结果
        print("\n🤝 步骤 7/9: 集成多算法结果")
        df_ensemble = ensemble_decision(
            y_true, cleanlab_suspects, cleanlab_scores,
            kmeans_suspects, kmeans_scores,
            isolation_suspects, isolation_scores, CONFIG["ENSEMBLE_CONFIG"]
        )
        print(f"   集成完成: {len(df_ensemble)} 个样本处理")

        # 8. 导出增强版结果
        print("\n💾 步骤 8/9: 导出结果")
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

        # 9. 生成可视化图表
        if CONFIG["ENABLE_VISUALIZATION"]:
            print("\n🎨 步骤 9/9: 生成可视化图表")
            try:
                visualizer = AnomalyDetectionVisualizer(CONFIG["VISUALIZATION_DIR"])
                visualizer.visualize_all(df_ensemble, img_paths)
                print(f"   可视化完成: 图表已保存至 {CONFIG['VISUALIZATION_DIR']}")
            except Exception as e:
                print(f"   ⚠️ 可视化生成失败: {e}")
                print("   程序将继续执行...")
        else:
            print("\n⏭️ 步骤 9/9: 跳过可视化（已禁用）")

        # 统计信息
        suspects_count = len(df_ensemble[df_ensemble['is_suspect'] == True])
        clean_count = len(df_ensemble[df_ensemble['is_suspect'] == False])

        print(f"\n📈 清洗统计:")
        print(f"   总样本数: {len(df_ensemble)}")
        print(f"   可疑样本: {suspects_count} ({suspects_count / len(df_ensemble) * 100:.1f}%)")
        print(f"   干净样本: {clean_count} ({clean_count / len(df_ensemble) * 100:.1f}%)")

        total_time = time.time() - start_time
        print(f"\n🎉 增强版清洗完成！总耗时: {total_time:.1f}秒")

    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断执行")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 清理日志系统
        cleanup_logging(dual_output, original_stdout, log_file)


if __name__ == "__main__":
    main()