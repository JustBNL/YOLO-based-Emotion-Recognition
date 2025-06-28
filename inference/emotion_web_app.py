import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import time
import threading
from queue import Queue
from typing import Optional, List
import torch

# 导入推理模块
try:
    from infer_images import load_models as load_image_models, CONFIG as IMAGE_CONFIG
    from infer_stream import load_models as load_stream_models, CONFIG as STREAM_CONFIG, FrameGrabber
except ImportError as e:
    print(f"❌ 导入推理模块失败: {e}")
    exit(1)


class OptimizedEmotionApp:
    def __init__(self):
        self.models_loaded = False
        self.det_model = None
        self.cls_model = None
        self.camera_active = False
        self.camera_thread = None
        self.current_frame = None

        # 性能优化参数
        self.frame_skip = 3
        self.frame_counter = 0
        self.last_detection_result = None
        self.cache_counter = 0
        self.cache_frames = 5

        # FPS统计
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def load_models_once(self):
        """延迟加载模型"""
        if not self.models_loaded:
            print("🔹 正在加载模型...")
            self.det_model, self.cls_model = load_image_models(IMAGE_CONFIG)
            self.models_loaded = True
            print("✅ 模型加载完成")

    def draw_label(self, img: np.ndarray, text: str, pos: tuple, color: tuple):
        """绘制标签"""
        font_scale = max(img.shape[1] / 800.0, 0.8)
        thickness = max(1, int(font_scale * 2))

        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, pos, (pos[0] + w + 10, pos[1] - h - 10), color, -1)
        cv2.putText(img, text, (pos[0] + 5, pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness + 1)

    def process_detection(self, img, config, use_batch=False):
        """统一的检测处理函数"""
        # 人脸检测
        det_res = self.det_model(img, imgsz=640, conf=config["conf_thres"],
                                 device=config["device"], verbose=False)[0]

        results = []
        faces = []
        coords = []

        # 收集人脸
        for box in det_res.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            if x2 <= x1 or y2 <= y1:
                continue
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
            faces.append(face)
            coords.append((x1, y1, x2, y2))

        if not faces:
            return img

        # 表情分类
        if use_batch and len(faces) > 1:
            # 批量处理
            batch = []
            for face in faces:
                face_resized = cv2.resize(face, (config["img_size"], config["img_size"]))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                batch.append(face_rgb.transpose(2, 0, 1) / 255.0)

            batch_tensor = torch.from_numpy(np.stack(batch)).float().to(config["device"])
            cls_results = self.cls_model.predict(batch_tensor, device=config["device"], verbose=False)

            # 处理批量结果
            for i, (cls_res, (x1, y1, x2, y2)) in enumerate(zip(cls_results, coords)):
                label, color = self._process_classification_result(cls_res, config)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                self.draw_label(img, label, (x1, y1), color)
        else:
            # 单个处理
            for face, (x1, y1, x2, y2) in zip(faces, coords):
                face_resized = cv2.resize(face, (config["img_size"], config["img_size"]))
                cls_res = self.cls_model(face_resized, imgsz=config["img_size"],
                                         device=config["device"], verbose=False)[0]

                label, color = self._process_classification_result(cls_res, config)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                self.draw_label(img, label, (x1, y1), color)

        return img

    def _process_classification_result(self, cls_res, config):
        """处理分类结果"""
        if hasattr(cls_res, 'probs') and cls_res.probs is not None:
            if config == IMAGE_CONFIG:
                # 图片模式
                class_bias = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
                logits = cls_res.probs.data.cpu().numpy()
                adjusted_logits = logits * class_bias
                probs = adjusted_logits / np.sum(adjusted_logits)
                idx = int(np.argmax(probs))
                prob = float(probs[idx])
            else:
                # 视频/摄像头模式
                class_bias = torch.tensor([1.4, 1.7, 1.5, 0.9, 1.3, 0.7, 1.6], device=config["device"])
                logits = cls_res.probs.data
                adjusted_logits = logits * class_bias
                adjusted_probs = torch.nn.functional.softmax(adjusted_logits, dim=0)
                idx = int(torch.argmax(adjusted_probs))
                prob = float(adjusted_probs[idx])
        else:
            idx, prob = 0, 0.0

        # 生成标签
        if config.get("label_display") == "code":
            label = f"{idx}"
        else:
            label = f"{config['names'][idx]} {prob:.2f}"

        color = config["colors"][idx % len(config["colors"])]
        return label, color

    def process_image(self, image):
        """处理单张图片"""
        if image is None:
            return None
        try:
            self.load_models_once()
            return self.process_detection(image.copy(), IMAGE_CONFIG)
        except Exception as e:
            print(f"❌ 图片处理失败: {e}")
            return image

    def process_images_batch(self, files):
        """批量处理图片"""
        if not files:
            return None

        results = []
        temp_dir = Path("temp_output/images")  # 修改为统一的临时目录
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 清理旧的临时文件
        for old_file in temp_dir.glob("*.jpg"):
            try:
                old_file.unlink()
            except:
                pass

        for i, file in enumerate(files):
            try:
                image = cv2.imread(file.name)
                if image is not None:
                    result = self.process_image(image)  # 直接处理BGR格式
                    if result is not None:
                        # 使用更规范的文件命名
                        temp_path = temp_dir / f"result_{i:03d}.jpg"
                        cv2.imwrite(str(temp_path), result)
                        results.append(str(temp_path))
            except Exception as e:
                print(f"❌ 处理文件失败 {file.name}: {e}")

        return results

    def process_video(self, video_file, progress=gr.Progress()):
        """处理视频文件"""
        if video_file is None:
            return None, "请先上传视频文件"

        try:
            self.load_models_once()

            output_dir = Path("temp_output")
            output_dir.mkdir(exist_ok=True)

            input_path = video_file.name
            output_path = output_dir / f"processed_{int(time.time())}.mp4"

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return None, "无法打开视频文件"

            # 获取视频属性
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self.process_detection(frame, STREAM_CONFIG, use_batch=True)
                out.write(processed_frame)

                frame_count += 1
                if frame_count % 10 == 0:
                    progress_value = frame_count / total_frames
                    progress(progress_value, f"处理进度: {frame_count}/{total_frames} 帧")

            cap.release()
            out.release()
            return str(output_path), f"视频处理完成！共处理 {frame_count} 帧"

        except Exception as e:
            print(f"❌ 视频处理失败: {e}")
            return None, f"视频处理失败: {str(e)}"

    def process_camera_frame(self, frame):
        """优化的摄像头帧处理"""
        try:
            self.frame_counter += 1

            # 跳帧逻辑
            should_detect = (
                    self.frame_counter % self.frame_skip == 0 or
                    self.last_detection_result is None or
                    self.cache_counter >= self.cache_frames
            )

            if should_detect:
                processed_frame = self.process_detection(frame, STREAM_CONFIG, use_batch=True)
                self.last_detection_result = processed_frame
                self.cache_counter = 0
            else:
                processed_frame = self.last_detection_result if self.last_detection_result is not None else frame
                self.cache_counter += 1

            # 添加性能信息
            self._update_fps()
            self._draw_performance_info(processed_frame)

            return processed_frame

        except Exception as e:
            print(f"❌ 摄像头帧处理失败: {e}")
            return frame

    def _update_fps(self):
        """更新FPS"""
        self.fps_counter += 1
        current_time = time.time()
        elapsed = current_time - self.fps_start_time

        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = current_time

    def _draw_performance_info(self, frame):
        """绘制性能信息"""
        try:
            info_text = f"FPS: {self.current_fps:.1f} | Skip: 1/{self.frame_skip} | Cache: {self.cache_counter}/{self.cache_frames}"

            # 半透明背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (400, 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # 绘制文本
            cv2.putText(frame, info_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        except Exception as e:
            print(f"⚠️ 绘制性能信息失败: {e}")

    def get_available_cameras(self):
        """获取可用摄像头"""
        cameras = []
        for i in range(5):  # 减少检测数量
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cameras.append(f"摄像头 {i}")
                cap.release()
        return cameras if cameras else ["无可用摄像头"]

    def start_camera(self, camera_choice, skip_frames=5):
        """启动摄像头"""
        if camera_choice == "无可用摄像头":
            return "没有可用的摄像头"

        if self.camera_active:
            self.stop_camera()
            time.sleep(0.5)

        try:
            camera_id = int(camera_choice.split()[-1])
            self.frame_skip = max(1, skip_frames)
            self._reset_camera_state()

            self.load_models_once()
            self.camera_active = True

            def camera_worker():
                try:
                    grabber = FrameGrabber(camera_id, queue_size=3)
                    grabber.start()

                    while self.camera_active:
                        ok, frame = grabber.read(timeout=1.0)
                        if not ok:
                            continue

                        processed_frame = self.process_camera_frame(frame)
                        if processed_frame is not None:
                            self.current_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                    grabber.stop()
                except Exception as e:
                    print(f"摄像头工作线程错误: {e}")
                    self.camera_active = False

            self.camera_thread = threading.Thread(target=camera_worker, daemon=True)
            self.camera_thread.start()
            return f"摄像头已启动: {camera_choice} (跳帧: 1/{self.frame_skip})"

        except Exception as e:
            self.camera_active = False
            return f"启动摄像头失败: {e}"

    def stop_camera(self):
        """停止摄像头"""
        self.camera_active = False
        if self.camera_thread:
            self.camera_thread.join(timeout=2.0)
        self._reset_camera_state()
        return "摄像头已停止"

    def _reset_camera_state(self):
        """重置摄像头状态"""
        self.current_frame = None
        self.frame_counter = 0
        self.last_detection_result = None
        self.cache_counter = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()

    def get_camera_frame(self):
        """获取摄像头帧"""
        return self.current_frame if self.camera_active else None

    def update_skip_frames(self, skip_frames):
        """更新跳帧设置"""
        self.frame_skip = max(1, skip_frames)
        return f"跳帧设置已更新: 1/{self.frame_skip}"


# 创建应用实例
app = OptimizedEmotionApp()


def create_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="情感识别系统", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎭 情感识别系统")

        with gr.Tabs():
            # 图片识别
            with gr.Tab("📸 图片识别"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.File(
                            label="上传图片",
                            file_types=["image"],
                            file_count="multiple"
                        )
                        image_btn = gr.Button("开始识别", variant="primary")

                    with gr.Column():
                        image_output = gr.Gallery(
                            label="识别结果",
                            columns=1,
                            height=None,
                            rows=6,
                            allow_preview=True,
                            preview=True,
                            show_download_button=True,
                            show_share_button=False,
                        )

                image_btn.click(
                    fn=app.process_images_batch,
                    inputs=image_input,
                    outputs=image_output,
                    show_progress=True
                )

            # 视频识别
            with gr.Tab("🎬 视频识别"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.File(
                            label="上传视频",
                            file_types=["video"]
                        )
                        video_btn = gr.Button("开始处理", variant="primary")

                    with gr.Column():
                        video_status = gr.Textbox(
                            label="处理状态",
                            value="等待上传视频...",
                            interactive=False
                        )
                        video_output = gr.Video(
                            label="处理结果",
                            height=400
                        )

                video_btn.click(
                    fn=app.process_video,
                    inputs=video_input,
                    outputs=[video_output, video_status],
                    show_progress=True
                )

            # 实时摄像头
            with gr.Tab("📹 实时摄像头"):
                with gr.Row():
                    with gr.Column(scale=2):
                        camera_dropdown = gr.Dropdown(
                            choices=app.get_available_cameras(),
                            value=app.get_available_cameras()[0],
                            label="选择摄像头"
                        )

                        skip_frames_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="跳帧设置",
                            info="数值越大性能越好"
                        )

                        with gr.Row():
                            start_btn = gr.Button("启动", variant="primary")
                            stop_btn = gr.Button("停止", variant="secondary")

                        update_skip_btn = gr.Button("更新跳帧", variant="secondary")

                        camera_status = gr.Textbox(
                            label="状态",
                            value="摄像头未启动",
                            interactive=False
                        )

                    with gr.Column(scale=2):
                        camera_output = gr.Image(
                            label="实时画面",
                            height=500
                        )

                # 事件绑定
                start_btn.click(
                    fn=lambda cam, skip: app.start_camera(cam, skip),
                    inputs=[camera_dropdown, skip_frames_slider],
                    outputs=camera_status
                )

                stop_btn.click(
                    fn=app.stop_camera,
                    outputs=camera_status
                )

                update_skip_btn.click(
                    fn=app.update_skip_frames,
                    inputs=skip_frames_slider,
                    outputs=camera_status
                )

                # 定时器更新摄像头画面
                camera_timer = gr.Timer(value=0.1)
                camera_timer.tick(
                    fn=app.get_camera_frame,
                    outputs=camera_output
                )

        # 使用说明
        with gr.Accordion("使用说明", open=False):
            gr.Markdown("""
            ### 🚀 性能优化功能
            - **跳帧处理**: 减少计算量，提升实时性能
            - **结果缓存**: 复用检测结果，降低CPU使用率
            - **批量处理**: 多人脸场景下的批量推理优化

            ### 支持的情感类别
            😠 生气 | 🤢 厌恶 | 😨 恐惧 | 😊 快乐 | 😢 悲伤 | 😐 中性 | 😲 惊讶
            """)

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        debug=True
    )