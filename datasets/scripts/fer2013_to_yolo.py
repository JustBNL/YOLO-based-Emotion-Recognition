import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# --- 配置参数 ---

# 1. fer2013.csv 文件的路径
csv_path = '../raw/fer2013.csv'

# 2. 输出数据集的根目录名称
output_dir = '../processed/fer2013_yolo'

# 3. 情绪标签到文件夹名称的映射
# YOLO 推荐使用数字和下划线开头的文件夹名，以保持类别顺序
emotion_labels = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6'
}

# 4. fer2013.csv中的'Usage'列到目标文件夹的映射
usage_mapping = {
    'Training': 'train',
    'PublicTest': 'val',  # PublicTest 通常用作验证集
    'PrivateTest': 'test'  # PrivateTest 通常用作测试集
}


# --- 脚本主逻辑 ---

def create_yolo_dataset(csv_path, output_dir):
    """
    读取 fer2013.csv 文件并将其转换为 YOLO 分类数据集格式。
    """
    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: CSV 文件未找到 at '{csv_path}'")
        return

    # 读取CSV文件
    print("正在读取 CSV 文件...")
    df = pd.read_csv(csv_path)

    print("开始创建目录结构...")
    # 创建主输出目录和子目录 (train, val, test)
    for usage in usage_mapping.values():
        usage_path = os.path.join(output_dir, usage)
        # 在每个子目录中为每个情绪创建文件夹
        for label_name in emotion_labels.values():
            class_path = os.path.join(usage_path, label_name)
            os.makedirs(class_path, exist_ok=True)

    print("开始处理图像并保存...")
    # 使用 tqdm 创建一个进度条
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="处理图像"):
        try:
            # 获取情绪、像素和用途
            emotion = int(row['emotion'])
            pixels_str = row['pixels']
            usage = row['Usage']

            # 将像素字符串转换为 48x48 的 numpy 数组 (图像)
            pixels = np.array(pixels_str.split(), 'uint8')
            image = pixels.reshape(48, 48)

            # 获取目标文件夹路径
            split_dir = usage_mapping.get(usage)
            if split_dir is None:
                print(f"警告: 在行 {index} 发现未知的 Usage '{usage}'，已跳过。")
                continue

            class_name = emotion_labels.get(emotion)
            if class_name is None:
                print(f"警告: 在行 {index} 发现未知的情绪标签 '{emotion}'，已跳过。")
                continue

            # 构建最终的文件保存路径和文件名
            # 文件名使用 "usage_emotion_index.png" 格式以确保唯一性
            image_filename = f"{split_dir}_{emotion}_{index}.png"
            image_path = os.path.join(output_dir, split_dir, class_name, image_filename)

            # 保存图像
            cv2.imwrite(image_path, image)

        except Exception as e:
            print(f"处理行 {index} 时发生错误: {e}")

    print("\n数据集转换完成！")
    print(f"数据集已保存在: '{os.path.abspath(output_dir)}'")
    print("\n目录结构预览:")
    for split in os.listdir(output_dir):
        split_path = os.path.join(output_dir, split)
        if os.path.isdir(split_path):
            print(f"  - {split}/")
            for class_dir in os.listdir(split_path)[:2]:  # 只显示前两个类别作为示例
                print(f"    - {class_dir}/")
            print("    - ...")


if __name__ == '__main__':
    create_yolo_dataset(csv_path, output_dir)