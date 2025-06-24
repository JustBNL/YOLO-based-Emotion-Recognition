#!/usr/bin/env python3
"""
recursively_delete_npv.py  ——  递归清理指定后缀文件（默认 .npv）

使用方法
---------
1. 修改下方 “配置区” 的 ROOT_DIR / EXT / DELETE。
2. 双击或 `python recursively_delete_npv.py` 运行。
   - 当 DELETE=False 时，仅打印将被删除的文件列表（安全预览）
   - 当 DELETE=True  时，真正执行删除
"""

# ────────── 配置区 ──────────
ROOT_DIR = r"D:\Document\PycharmProjects\YOLO-based-Emotion-Recognition\datasets\cls\processed\affectnet-clean"  # 目标目录 (绝对或相对路径都可)
EXT      = ".npy"                          # 要删除的文件扩展名
DELETE   = True                           # True: 实删  |  False: 预览
# ──────────────────────────

from pathlib import Path
import sys

def find_files(root: Path, ext: str):
    """递归生成 root 下后缀为 ext 的文件路径"""
    return (p for p in root.rglob(f"*{ext}") if p.is_file())

def main():
    root = Path(ROOT_DIR).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"❌ 目标路径不存在或不是文件夹: {root}")

    targets = list(find_files(root, EXT))

    if not targets:
        print(f"✅ 目录 {root} 下未找到 {EXT} 文件。")
        return

    if DELETE:
        for f in targets:
            try:
                f.unlink()
                print(f"🗑️  Deleted {f}")
            except Exception as e:
                print(f"⚠️  Failed to delete {f}: {e}")
        print(f"🎉 完成！共删除 {len(targets)} 个文件。")
    else:
        print("以下文件将被删除（预览模式，修改 DELETE=True 后会真正删除）：\n")
        for f in targets:
            print(f)
        print(f"\n共计 {len(targets)} 个文件。")

if __name__ == "__main__":
    main()
