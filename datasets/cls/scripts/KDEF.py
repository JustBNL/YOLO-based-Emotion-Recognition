import os
import shutil
from tqdm import tqdm

# 配置文件路径
kdef_root_dir = "..\\raw\\KDEF"
output_root_dir = "..\\processed\\KDEF"

# 定义表情映射
expression_mapping = {
    "AF": "fear",
    "AN": "angry",
    "DI": "disgust",
    "HA": "happy",
    "NE": "neutral",
    "SA": "sad",
    "SU": "surprise"
}


def convert_kdef_to_class_folders(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建所有表情类别文件夹
    for emotion in set(expression_mapping.values()):
        os.makedirs(os.path.join(output_dir, emotion), exist_ok=True)

    # 收集所有图片路径
    all_images = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                all_images.append(os.path.join(root, file))

    print(f"Found {len(all_images)} images total")

    # 处理每张图片
    copied_count = 0
    for img_path in tqdm(all_images, desc="Processing images"):
        img_name = os.path.basename(img_path)

        # 去除扩展名并转为大写，统一处理
        base_name = os.path.splitext(img_name)[0].upper()

        # 验证文件名格式（至少6个字符）
        if len(base_name) < 6:
            continue

        try:
            # 提取关键部分 - 更灵活的方式
            # 表情代码：第5-6个字符（索引4:6）
            expression_code = base_name[4:6]

            # 角度代码：尝试从第7-8个字符提取，如果不够则从结尾提取
            angle_code = base_name[6:8] if len(base_name) >= 8 else base_name[-1] + " "
        except IndexError:
            continue

        # 只处理正脸图片 (S或S后跟空格)
        if angle_code.strip() != 'S':
            continue

        # 获取表情类别
        emotion = expression_mapping.get(expression_code)
        if not emotion:
            continue

        # 目标路径
        target_dir = os.path.join(output_dir, emotion)
        target_path = os.path.join(target_dir, img_name)

        # 复制图片
        shutil.copy2(img_path, target_path)
        copied_count += 1

    print(f"Successfully copied {copied_count} frontal face images")


def main():
    convert_kdef_to_class_folders(kdef_root_dir, output_root_dir)
    print("Processing completed!")


if __name__ == "__main__":
    main()