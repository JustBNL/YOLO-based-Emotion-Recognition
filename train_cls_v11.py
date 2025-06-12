# train_cls_v11.py  —— 请整段覆盖
from ultralytics import YOLO
from pathlib import Path

def main():
    model = YOLO("yolo11n-cls.pt")                          # 记得是 -cls 权重

    model.train(
        data=str(r"D:\Document\PycharmProjects\YOLO-based-Emotion-Recognition\fer2013_yolo"),
        imgsz=128,
        batch=512,
        epochs=200,
        device=0,
        amp=True,
    )

if __name__ == "__main__":
    main()
