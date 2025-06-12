from yolo_train_utils import run_training

class YOLOv12Trainer:
    def __init__(self):
        self.model_tag = "yolov12"
        self.weight_path = "yolo12n.pt"

    def train(self, epochs=50, imgsz=64, batch=16, device=0, extra_train_args=None):
        print("========== YOLOv12 训练 ==========")
        return run_training(
            self.model_tag,
            self.weight_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            extra_train_args=extra_train_args,
        )

if __name__ == '__main__':
    trainer = YOLOv12Trainer()
    trainer.train()
