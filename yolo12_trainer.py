# yolo12_trainer.py
from yolo_train_utils import run_training


# =================================================================
# 统一超参数（只改这里即可）
# =================================================================
TRAIN_CFG = {
    # -------- 基础 --------
    'imgsz': 160,
    'batch': 4,
    'epochs': 10,
    'nbs': 16,              # = 累积 4
    # -------- 优化器 --------
    'optimizer': 'AdamW',
    'lr0': 0.003,
    'lrf': 0.1,
    'cos_lr': True,
    'warmup_epochs': 3.0,
    'patience': 30,
    'cls': 2.0,             # 分类权重↑
    # -------- 轻量增强 --------
    'mosaic': 0.0, 'mixup': 0.0, 'copy_paste': 0.0,
    'degrees': 10.0, 'translate': 0.05, 'scale': 0.4,
    'fliplr': 0.5, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
    # -------- 其它 --------
    'freeze': [0, 1, 2],    # 先冻 10 轮
    'close_mosaic': 10,
    'workers': 4,
    'seed': 42,
    # 'anchors': 'rafdb9.npy',  # 若已聚类
    'name': 'yolov12_3',   # 避免加载历史 last.pt
    'resume': False,
}


class YOLOv12Trainer:
    """简单封装，所有参数通过 extra_train_args 透传"""

    def __init__(self,
                 model_tag: str = "yolov12v2",
                 weight_path: str = "yolov12n.pt"):
        self.model_tag = model_tag
        self.weight_path = weight_path

    def train(self,
              device: int = 0,
              extra_train_args: dict | None = None):
        print("\n========== YOLOv12 训练启动 ==========")
        return run_training(
            model_tag=self.model_tag,
            weight_path=self.weight_path,
            # 这些默认值会被 extra_train_args 覆盖
            epochs=50,
            imgsz=64,
            batch=16,
            device=device,
            extra_train_args=extra_train_args,
        )


if __name__ == '__main__':
    # 单行开跑
    YOLOv12Trainer().train(device=0, extra_train_args=TRAIN_CFG)
