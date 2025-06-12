# yolo12_trainer.py
from yolo_train_utils import run_training


# =================================================================
# 统一超参数（只改这里即可）
# =================================================================
# yolo12_trainer_best.py  —— 只粘贴到你现有脚本即可
TRAIN_CFG = dict(
    # ---- 基础 ----
    imgsz=128,
    batch=256,
    epochs=50,
    # ---- 资源相关 ----
    cache='disk',
    workers=4,
    amp=True,
    # ---- 其余保持原设置 ----
    optimizer='SGD', lr0=0.003, lrf=0.10,
    momentum=0.937, weight_decay=5e-4,
    nbs=64, cos_lr=True,
    mosaic=1.0, mixup=0.15, close_mosaic=20,
    cls=0.7,
    seed=0, resume=False,
    name='yolov12_rafdb_stage1_safe1'
)

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
