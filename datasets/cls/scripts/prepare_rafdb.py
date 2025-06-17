from pathlib import Path
import shutil
import argparse
from tqdm import tqdm

# --------- 可按需修改 ----------
SRC_ROOT  = Path('RAF')        # 原始 RAF 根目录
DST_ROOT  = Path('RAF_flat')   # 输出根目录
CLASSES   = [str(i) for i in range(7)]  # 0~6
SPLITS    = ['train', 'valid']
# --------------------------------


def flatten_dataset(src_root: Path, dst_root: Path) -> None:
    """将 src_root 中的分类子目录扁平化复制到 dst_root 并生成 YOLO 标签"""
    if not src_root.exists():
        raise FileNotFoundError(f'❌ 未找到原始数据根目录: {src_root.resolve()}')

    # 创建目标目录结构
    for split in SPLITS:
        (dst_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dst_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        print(f'✅ {split}: 处理中 …')
        idx = 0  # 全局计数避免重名
        for cls in CLASSES:                                        # 0~6
            src_cls_dir = src_root / split / cls
            if not src_cls_dir.exists():
                print(f'  ⚠️ 跳过不存在的类别文件夹: {src_cls_dir}')
                continue

            for img_path in tqdm(list(src_cls_dir.glob('*.*')),
                                  desc=f'   class {cls}',
                                  leave=False):
                if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    continue

                # 生成唯一文件名：cls_序号.ext
                new_name = f'{cls}_{idx:06d}{img_path.suffix.lower()}'
                idx += 1

                # 拷贝图片
                dst_img = dst_root / 'images' / split / new_name
                shutil.copy2(img_path, dst_img)

                # 写标签
                dst_lbl = dst_root / 'labels' / split / (new_name.rsplit('.', 1)[0] + '.txt')
                dst_lbl.write_text(f'{cls} 0.5 0.5 1.0 1.0\n')

        print(f'   👉 已处理 {idx} 张图片')

    # 统计
    train_cnt = sum(1 for _ in (dst_root / 'images/train').iterdir())
    valid_cnt = sum(1 for _ in (dst_root / 'images/valid').iterdir())
    print('\n🎉 数据集转换完成！')
    print(f'   images/train: {train_cnt} 张')
    print(f'   images/valid: {valid_cnt} 张')
    print(f'📁 新数据路径: {dst_root.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flatten RAF dataset for YOLO')
    parser.add_argument('--src',  type=Path, default=SRC_ROOT, help='原始 RAF 根目录')
    parser.add_argument('--dst',  type=Path, default=DST_ROOT, help='输出目录 (RAF_flat)')
    args = parser.parse_args()

    flatten_dataset(args.src, args.dst)