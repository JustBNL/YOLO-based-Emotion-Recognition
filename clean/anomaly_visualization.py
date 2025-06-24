import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AnomalyDetectionVisualizer:
    """异常检测结果可视化类"""

    def __init__(self, output_dir: Path = None):
        """
        初始化可视化器

        Args:
            output_dir: 输出目录，如果为None则使用当前目录下的visualizations文件夹
        """
        if output_dir is None:
            self.output_dir = Path("visualizations")
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📊 可视化输出目录: {self.output_dir}")

    def plot_detection_overview(self, df_results: pd.DataFrame, save_name: str = "detection_overview"):
        """绘制检测结果总览"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('异常检测结果总览', fontsize=16, fontweight='bold')

        # 1. 各算法检测数量对比
        detection_counts = {
            'CleanLab': df_results['cleanlab_suspect'].sum(),
            'K-Means': df_results['kmeans_suspect'].sum(),
            'Isolation Forest': df_results['isolation_suspect'].sum(),
            '集成结果': df_results['is_suspect'].sum()
        }

        axes[0, 0].bar(detection_counts.keys(), detection_counts.values(),
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        axes[0, 0].set_title('各算法检测数量对比')
        axes[0, 0].set_ylabel('检测到的异常样本数')
        for i, (k, v) in enumerate(detection_counts.items()):
            axes[0, 0].text(i, v + 10, str(v), ha='center', va='bottom')

        # 2. 检测比例饼图
        total_samples = len(df_results)
        suspect_count = df_results['is_suspect'].sum()
        clean_count = total_samples - suspect_count

        axes[0, 1].pie([suspect_count, clean_count],
                       labels=[f'可疑样本\n({suspect_count})', f'正常样本\n({clean_count})'],
                       colors=['#FF6B6B', '#96CEB4'], autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('样本分布')

        # 3. 投票数分布
        vote_counts = df_results['vote_count'].value_counts().sort_index()
        axes[1, 0].bar(vote_counts.index, vote_counts.values, color='#45B7D1')
        axes[1, 0].set_title('投票数分布')
        axes[1, 0].set_xlabel('投票数')
        axes[1, 0].set_ylabel('样本数量')
        for i, v in enumerate(vote_counts.values):
            axes[1, 0].text(vote_counts.index[i], v + 5, str(v), ha='center', va='bottom')

        # 4. 综合分数分布
        axes[1, 1].hist(df_results['composite_score'], bins=50, alpha=0.7, color='#4ECDC4')
        axes[1, 1].axvline(df_results['composite_score'].median(), color='red',
                           linestyle='--', label=f'中位数: {df_results["composite_score"].median():.3f}')
        axes[1, 1].set_title('综合质量分数分布')
        axes[1, 1].set_xlabel('综合分数')
        axes[1, 1].set_ylabel('样本数量')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存: {save_name}.png")

    def plot_score_distributions(self, df_results: pd.DataFrame, save_name: str = "score_distributions"):
        """绘制各算法分数分布"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('各算法质量分数分布', fontsize=16, fontweight='bold')

        # 准备数据
        algorithms = ['cleanlab', 'kmeans', 'isolation', 'composite']
        titles = ['CleanLab分数', 'K-Means分数', 'Isolation Forest分数', '综合分数']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        for i, (alg, title, color) in enumerate(zip(algorithms, titles, colors)):
            row, col = i // 2, i % 2
            score_col = f'{alg}_score'

            # 分别绘制可疑和正常样本的分数分布
            suspect_scores = df_results[df_results['is_suspect'] == True][score_col]
            normal_scores = df_results[df_results['is_suspect'] == False][score_col]

            axes[row, col].hist(normal_scores, bins=30, alpha=0.7, label='正常样本', color=color)
            axes[row, col].hist(suspect_scores, bins=30, alpha=0.7, label='可疑样本', color='red')

            axes[row, col].set_title(f'{title}分布')
            axes[row, col].set_xlabel('分数')
            axes[row, col].set_ylabel('样本数量')
            axes[row, col].legend()

            # 添加统计信息
            axes[row, col].axvline(normal_scores.mean(), color=color, linestyle='--', alpha=0.8)
            axes[row, col].axvline(suspect_scores.mean(), color='red', linestyle='--', alpha=0.8)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存: {save_name}.png")

    def plot_algorithm_correlation(self, df_results: pd.DataFrame, save_name: str = "algorithm_correlation"):
        """绘制算法间相关性分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('算法间相关性分析', fontsize=16, fontweight='bold')

        # 1. 算法检测结果重叠热力图
        overlap_matrix = np.zeros((3, 3))
        algorithms = ['cleanlab_suspect', 'kmeans_suspect', 'isolation_suspect']
        alg_names = ['CleanLab', 'K-Means', 'Isolation']

        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms):
                if i == j:
                    overlap_matrix[i, j] = df_results[alg1].sum()
                else:
                    overlap_matrix[i, j] = ((df_results[alg1] == True) & (df_results[alg2] == True)).sum()

        sns.heatmap(overlap_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                    xticklabels=alg_names, yticklabels=alg_names, ax=axes[0, 0])
        axes[0, 0].set_title('算法检测结果重叠矩阵')

        # 2. 分数相关性热力图
        score_cols = ['cleanlab_score', 'kmeans_score', 'isolation_score', 'composite_score']
        correlation_matrix = df_results[score_cols].corr()

        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                    xticklabels=['CleanLab', 'K-Means', 'Isolation', 'Composite'],
                    yticklabels=['CleanLab', 'K-Means', 'Isolation', 'Composite'],
                    ax=axes[0, 1])
        axes[0, 1].set_title('分数相关性矩阵')

        # 3. 散点图：CleanLab vs K-Means
        suspect_mask = df_results['is_suspect'] == True
        axes[1, 0].scatter(df_results[~suspect_mask]['cleanlab_score'],
                           df_results[~suspect_mask]['kmeans_score'],
                           alpha=0.6, c='green', label='正常样本', s=20)
        axes[1, 0].scatter(df_results[suspect_mask]['cleanlab_score'],
                           df_results[suspect_mask]['kmeans_score'],
                           alpha=0.6, c='red', label='可疑样本', s=20)
        axes[1, 0].set_xlabel('CleanLab分数')
        axes[1, 0].set_ylabel('K-Means分数')
        axes[1, 0].set_title('CleanLab vs K-Means分数')
        axes[1, 0].legend()

        # 4. 散点图：综合分数 vs 投票数
        for vote_count in sorted(df_results['vote_count'].unique()):
            subset = df_results[df_results['vote_count'] == vote_count]
            axes[1, 1].scatter(subset['composite_score'], [vote_count] * len(subset),
                               alpha=0.6, label=f'投票数={vote_count}', s=20)

        axes[1, 1].set_xlabel('综合分数')
        axes[1, 1].set_ylabel('投票数')
        axes[1, 1].set_title('综合分数 vs 投票数')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存: {save_name}.png")

    def plot_class_wise_analysis(self, df_results: pd.DataFrame, save_name: str = "class_wise_analysis"):
        """绘制按类别的异常检测分析"""
        # 按类别统计
        class_stats = df_results.groupby('true_label').agg({
            'is_suspect': ['count', 'sum'],
            'composite_score': ['mean', 'std'],
            'vote_count': 'mean'
        }).round(3)

        class_stats.columns = ['总数', '可疑数', '平均分数', '分数标准差', '平均投票数']
        class_stats['可疑比例'] = (class_stats['可疑数'] / class_stats['总数'] * 100).round(1)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('按类别的异常检测分析', fontsize=16, fontweight='bold')

        # 1. 各类别可疑样本数量
        axes[0, 0].bar(class_stats.index, class_stats['可疑数'], color='#FF6B6B')
        axes[0, 0].set_title('各类别可疑样本数量')
        axes[0, 0].set_xlabel('类别')
        axes[0, 0].set_ylabel('可疑样本数')

        # 2. 各类别可疑比例
        axes[0, 1].bar(class_stats.index, class_stats['可疑比例'], color='#4ECDC4')
        axes[0, 1].set_title('各类别可疑比例 (%)')
        axes[0, 1].set_xlabel('类别')
        axes[0, 1].set_ylabel('可疑比例 (%)')

        # 3. 各类别平均分数
        axes[1, 0].bar(class_stats.index, class_stats['平均分数'], color='#45B7D1')
        axes[1, 0].set_title('各类别平均质量分数')
        axes[1, 0].set_xlabel('类别')
        axes[1, 0].set_ylabel('平均分数')

        # 4. 各类别样本总数
        axes[1, 1].bar(class_stats.index, class_stats['总数'], color='#96CEB4')
        axes[1, 1].set_title('各类别样本总数')
        axes[1, 1].set_xlabel('类别')
        axes[1, 1].set_ylabel('样本数')

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存: {save_name}.png")

        # 保存统计表格
        class_stats.to_csv(self.output_dir / f"{save_name}_stats.csv", encoding='utf-8')
        print(f"✅ 保存: {save_name}_stats.csv")

    def plot_top_suspects(self, df_results: pd.DataFrame, img_paths: list = None,
                          top_n: int = 20, save_name: str = "top_suspects"):
        """绘制最可疑样本的分析图"""
        top_suspects = df_results.nsmallest(top_n, 'composite_score')

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Top {top_n} 最可疑样本分析', fontsize=16, fontweight='bold')

        # 1. 最可疑样本的综合分数
        axes[0, 0].barh(range(len(top_suspects)), top_suspects['composite_score'], color='#FF6B6B')
        axes[0, 0].set_title('最可疑样本综合分数')
        axes[0, 0].set_xlabel('综合分数')
        axes[0, 0].set_ylabel('样本索引')
        axes[0, 0].set_yticks(range(len(top_suspects)))
        axes[0, 0].set_yticklabels(top_suspects['index'].values)

        # 2. 最可疑样本的投票分布
        vote_dist = top_suspects['vote_count'].value_counts().sort_index()
        axes[0, 1].bar(vote_dist.index, vote_dist.values, color='#4ECDC4')
        axes[0, 1].set_title('最可疑样本投票分布')
        axes[0, 1].set_xlabel('投票数')
        axes[0, 1].set_ylabel('样本数量')

        # 3. 最可疑样本的类别分布
        class_dist = top_suspects['true_label'].value_counts()
        axes[1, 0].bar(class_dist.index, class_dist.values, color='#45B7D1')
        axes[1, 0].set_title('最可疑样本类别分布')
        axes[1, 0].set_xlabel('类别')
        axes[1, 0].set_ylabel('样本数量')

        # 4. 各算法对最可疑样本的检测情况
        detection_stats = {
            'CleanLab': top_suspects['cleanlab_suspect'].sum(),
            'K-Means': top_suspects['kmeans_suspect'].sum(),
            'Isolation': top_suspects['isolation_suspect'].sum()
        }
        axes[1, 1].bar(detection_stats.keys(), detection_stats.values(), color='#96CEB4')
        axes[1, 1].set_title('各算法对最可疑样本的检测')
        axes[1, 1].set_ylabel('检测到的样本数')

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 保存: {save_name}.png")

        # 保存最可疑样本详细信息
        top_suspects_export = top_suspects[['index', 'true_label', 'composite_score', 'vote_count',
                                            'cleanlab_suspect', 'kmeans_suspect', 'isolation_suspect']]
        top_suspects_export.to_csv(self.output_dir / f"{save_name}_details.csv", index=False, encoding='utf-8')
        print(f"✅ 保存: {save_name}_details.csv")

    def generate_summary_report(self, df_results: pd.DataFrame):
        """生成汇总报告"""
        total_samples = len(df_results)
        suspect_count = df_results['is_suspect'].sum()

        report = f"""
# 异常检测结果汇总报告

## 基本统计
- 总样本数: {total_samples:,}
- 可疑样本数: {suspect_count:,}
- 可疑比例: {suspect_count / total_samples * 100:.2f}%
- 正常样本数: {total_samples - suspect_count:,}

## 各算法检测结果
- CleanLab检测: {df_results['cleanlab_suspect'].sum():,} 个 ({df_results['cleanlab_suspect'].sum() / total_samples * 100:.2f}%)
- K-Means检测: {df_results['kmeans_suspect'].sum():,} 个 ({df_results['kmeans_suspect'].sum() / total_samples * 100:.2f}%)
- Isolation Forest检测: {df_results['isolation_suspect'].sum():,} 个 ({df_results['isolation_suspect'].sum() / total_samples * 100:.2f}%)

## 分数统计
- 综合分数均值: {df_results['composite_score'].mean():.4f}
- 综合分数标准差: {df_results['composite_score'].std():.4f}
- 可疑样本平均分数: {df_results[df_results['is_suspect']]['composite_score'].mean():.4f}
- 正常样本平均分数: {df_results[~df_results['is_suspect']]['composite_score'].mean():.4f}

## 投票分析
- 3票可疑: {(df_results['vote_count'] == 3).sum():,} 个
- 2票可疑: {(df_results['vote_count'] == 2).sum():,} 个  
- 1票可疑: {(df_results['vote_count'] == 1).sum():,} 个
- 0票可疑: {(df_results['vote_count'] == 0).sum():,} 个

## 按类别统计
{df_results.groupby('true_label')['is_suspect'].agg(['count', 'sum', lambda x: x.sum() / len(x) * 100]).round(2)}
"""

        with open(self.output_dir / "summary_report.md", 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ 保存: summary_report.md")

    def visualize_all(self, df_results: pd.DataFrame, img_paths: list = None):
        """生成所有可视化图表"""
        print("🎨 开始生成可视化图表...")

        # 基础检查
        if df_results is None or len(df_results) == 0:
            print("❌ 数据为空，无法生成可视化")
            return

        try:
            # 1. 检测结果总览
            self.plot_detection_overview(df_results)

            # 2. 分数分布
            self.plot_score_distributions(df_results)

            # 3. 算法相关性
            self.plot_algorithm_correlation(df_results)

            # 4. 按类别分析
            self.plot_class_wise_analysis(df_results)

            # 5. 最可疑样本
            self.plot_top_suspects(df_results, img_paths)

            # 6. 汇总报告
            self.generate_summary_report(df_results)

            print(f"🎉 所有可视化图表已生成完成！")
            print(f"📁 输出目录: {self.output_dir}")

        except Exception as e:
            print(f"❌ 可视化生成失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    """测试函数"""
    # 创建示例数据进行测试
    n_samples = 1000
    n_classes = 8

    # 模拟数据
    np.random.seed(42)
    df_test = pd.DataFrame({
        'index': range(n_samples),
        'true_label': np.random.randint(0, n_classes, n_samples),
        'cleanlab_suspect': np.random.random(n_samples) < 0.1,
        'kmeans_suspect': np.random.random(n_samples) < 0.08,
        'isolation_suspect': np.random.random(n_samples) < 0.12,
        'cleanlab_score': np.random.beta(2, 2, n_samples),
        'kmeans_score': np.random.beta(2, 2, n_samples),
        'isolation_score': np.random.beta(2, 2, n_samples),
    })

    # 计算其他字段
    df_test['vote_count'] = (df_test['cleanlab_suspect'].astype(int) +
                             df_test['kmeans_suspect'].astype(int) +
                             df_test['isolation_suspect'].astype(int))

    df_test['composite_score'] = (df_test['cleanlab_score'] * 0.5 +
                                  df_test['kmeans_score'] * 0.3 +
                                  df_test['isolation_score'] * 0.2)

    df_test['is_suspect'] = (df_test['vote_count'] >= 2) | (df_test['composite_score'] <= 0.4)

    # 创建可视化器并生成图表
    visualizer = AnomalyDetectionVisualizer("test_visualizations")
    visualizer.visualize_all(df_test)


if __name__ == "__main__":
    main()