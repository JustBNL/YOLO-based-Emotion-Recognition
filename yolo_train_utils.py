import sys, os, time, threading, shutil, contextlib
from pathlib import Path
import torch, GPUtil, psutil

from ultralytics import YOLO   # 始终只用本地包

# ==================== 数据与路径相关配置 ====================
DATA_DIR = Path("./RAF_flat")          # 平铺后的数据集目录
YAML     = Path("data.yaml")           # YOLO 数据配置
PROJECT  = "runs/train"
WEIGHTS_DIR = Path(PROJECT)
LOG_DIR     = WEIGHTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def verify_dataset():
    """检查 train/valid 两个 split 的图片与标签数量"""
    for split in ("train", "valid"):
        p_img = DATA_DIR / "images" / split
        p_lbl = DATA_DIR / "labels" / split
        if not p_img.exists() or not p_lbl.exists():
            sys.exit(f"❌ 缺少 {p_img} 或 {p_lbl}")
        n_img = len(list(p_img.glob("*.jpg"))) + len(list(p_img.glob("*.png")))
        n_lbl = len(list(p_lbl.glob("*.txt")))
        print(f"✅ {split:5} 图像数: {n_img}, 标签数: {n_lbl}")
    if not YAML.exists():
        sys.exit(f"❌ 缺少 {YAML}，请检查路径")

def monitor(interval=30):
    """后台线程：定时打印 GPU/CPU/RAM 使用情况"""
    while True:
        try:
            g  = GPUtil.getGPUs()[0]
            ram = psutil.virtual_memory().percent
            print(f"\n🟢 GPU {g.memoryUsed/g.memoryTotal*100:4.1f}% | "
                  f"🌡️ {g.temperature}°C | 🧠 RAM {ram:4.1f}%")
        except Exception:
            pass
        time.sleep(interval)

@contextlib.contextmanager
def log_stdout(log_path: Path):
    """同时把 stdout 写入文件"""
    class Tee:
        def __init__(self, *files): self.files = files
        def write(self, obj): [f.write(obj) for f in self.files]
        def flush(self):       [f.flush()   for f in self.files]
    with open(log_path, "a", encoding="utf-8") as f:
        orig = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        try:   yield
        finally: sys.stdout = orig

def save_eval_plots(exp_dir: Path):
    """复制混淆矩阵 / PR / ROC 图到实验根目录"""
    for name in ["confusion_matrix.png", "pr_curve.png", "roc_curve.png"]:
        src = exp_dir / "results" / name
        if src.exists():
            shutil.copy(src, exp_dir)
            print(f"📊 已保存: {exp_dir/name}")

def run_training(
        model_tag: str,  # 实验名称/模型名称
        weight_path: str,  # 初始权重文件路径。比如 'yolov12n.pt'，用于初始化模型（不是断点恢复时）
        epochs=50,  # 训练的总轮数。决定模型将会遍历训练集多少次
        imgsz=64,  # 输入图片尺寸。YOLO会把图片缩放到这个尺寸（建议根据模型和显存调整）
        batch=16,  # 批次大小。一次送入模型训练的图片数量，显存大可以设大
        device=0,  # 训练所用GPU编号。0表示用第一块GPU，-1表示用CPU（不建议）
        extra_train_args=None):  # 其它可选的训练参数（字典类型），可以用来覆盖/补充默认超参数

    verify_dataset()
    threading.Thread(target=monitor, daemon=True).start()

    exp_name = f"{model_tag}_fer"
    exp_dir  = WEIGHTS_DIR / exp_name
    resume_pt = exp_dir / "weights/last.pt"
    log_path  = LOG_DIR / f"{exp_name}_log.txt"

    print(f"[info] 使用本地 ultralytics 包")
    print(f"[调试] YOLO类: {YOLO}, 路径: {YOLO.__module__}")

    # 断点续训 or 加载预训练
    model = YOLO(str(resume_pt)) if resume_pt.exists() else YOLO(weight_path)

    # 训练参数
    train_args = dict(
        data=str(YAML),  # 数据集的配置文件路径（data.yaml），包含train/val路径和类别信息
        epochs=epochs,  # 训练轮数（遍历整个训练集的次数）
        imgsz=imgsz,  # 输入图片的缩放尺寸（比如64，所有图片会缩放到64x64再送入网络）
        batch=batch,  # 批处理大小（一次送入多少张图片，受GPU显存影响）
        device=device,  # 用哪个GPU（如0表示第0块GPU，-1用CPU）
        project=PROJECT,  # 训练结果保存的根目录（如runs/train）
        name=exp_name,  # 本次实验的名字，作为保存子目录（如yolov12n.pt_fer）
        exist_ok=True,  # 如果保存目录已存在，不报错，直接覆盖
        resume=resume_pt.exists(),  # 断点续训，如果有断点权重last.pt就接着训
        amp=True,  # 是否开启混合精度（自动混合精度训练，节省显存，建议开启）
        patience=10,  # Early stopping耐心值。多少轮指标没提升就停止训练
        verbose=True,  # 详细日志输出
        save=True,  # 是否保存模型权重
        save_period=1,  # 每隔多少轮保存一次权重
        plots=True,  # 是否生成训练过程的可视化图表（loss、准确率、混淆矩阵等）
    )

    if extra_train_args: train_args.update(extra_train_args)

    with log_stdout(log_path):
        try:
            print("="*60)
            print(f"📅 开始训练: {model_tag} 目录: {exp_dir}")
            print(f"🔧 超参数: {train_args}")

            results = model.train(**train_args)

            print(f"\n✅ 训练完成！最佳权重: {exp_dir/'weights/best.pt'}")
            print("📊 评估中...")
            model.val(data=str(YAML), imgsz=imgsz, batch=batch, device=device,
                      save_json=True, plots=True, project=str(exp_dir), name="results")
            save_eval_plots(exp_dir)
            return exp_dir / "weights" / "best.pt"

        except KeyboardInterrupt:
            print("\n⏹ 训练被中断，未保存权重")
        except Exception as e:
            print(f"\n❌ 训练出错: {e}")
        finally:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            print("🧹 已清理CUDA缓存")
            print(f"📜 日志保存在: {log_path}")