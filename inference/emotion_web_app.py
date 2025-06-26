import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time
import threading
from queue import Queue, Empty
from typing import Optional, List, Union
import torch

# 导入推理模块
try:
    from infer_images import load_models as load_image_models, CONFIG as IMAGE_CONFIG
    from infer_stream import load_models as load_stream_models, CONFIG as STREAM_CONFIG, FrameGrabber
except ImportError as e:
    print(f"❌ 导入推理模块失败: {e}")
    print("请确保 infer_images.py 和 infer_stream.py 在同一目录下")
    exit(1)


class SimpleEmotionApp:
    def __init__(self):
        self.models_loaded = False
        self.det_model = None
        self.cls_model = None
        self.camera_active = False
        self.camera_thread = None
        self.frame_queue = Queue(maxsize=5)
        self.current_frame = None

        # 🚀 跳帧优化参数
        self.frame_skip = 3  # 每3帧处理1帧
        self.frame_counter = 0
        self.last_detection_result = None  # 缓存上次检测结果
        self.detection_cache_frames = 5  # 检测结果缓存帧数
        self.cache_counter = 0

        # 性能统计
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def load_models_once(self):
        """延迟加载模型，只加载一次"""
        if not self.models_loaded:
            print("🔹 正在加载模型...")
            # 使用相同的模型配置
            self.det_model, self.cls_model = load_image_models(IMAGE_CONFIG)
            self.models_loaded = True
            print("✅ 模型加载完成")

    def draw_label(self, img: np.ndarray, text: str, tl: tuple, color: tuple):
        """绘制带背景的标签"""
        font_scale = max(img.shape[1] / 800.0, 0.8)
        thickness = max(1, int(font_scale * 2))
        font = cv2.FONT_HERSHEY_SIMPLEX

        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        top_left = tl
        bottom_right = (tl[0] + w + 10, tl[1] - h - 10)

        cv2.rectangle(img, top_left, bottom_right, color, -1, cv2.LINE_AA)
        cv2.putText(img, text, (tl[0] + 5, tl[1] - 5), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)

    def process_image(self, image):
        """处理单张图片 - 实现与 infer_images.py 相同的效果"""
        if image is None:
            return None

        try:
            self.load_models_once()
            img = image.copy()

            # 人脸检测
            det_res = self.det_model(img, imgsz=640, conf=IMAGE_CONFIG["conf_thres"],
                                     device=IMAGE_CONFIG["device"], verbose=False)[0]

            # 处理每个检测到的人脸
            for box in det_res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                face = img[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # 调整人脸尺寸用于分类
                face_resized = cv2.resize(face, (IMAGE_CONFIG["img_size"], IMAGE_CONFIG["img_size"]))

                # 表情分类
                cls_res = self.cls_model(face_resized, imgsz=IMAGE_CONFIG["img_size"],
                                         device=IMAGE_CONFIG["device"], verbose=False)[0]

                class_bias = np.array([1, 1, 1, 1, 0.5, 1, 1], dtype=np.float32)
                logits = cls_res.probs.data.cpu().numpy()
                adjusted_logits = logits * class_bias
                probs = adjusted_logits / np.sum(adjusted_logits)

                idx = int(np.argmax(probs))
                prob = float(probs[idx])

                # 生成标签
                if IMAGE_CONFIG.get("label_display") == "code":
                    label = f"{idx}"
                else:
                    label = f"{IMAGE_CONFIG['names'][idx]} {prob:.2f}"

                # 绘制检测框和标签
                color = IMAGE_CONFIG["colors"][idx % len(IMAGE_CONFIG["colors"])]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                self.draw_label(img, label, (x1, y1), color)

            return img

        except Exception as e:
            print(f"❌ 图片处理失败: {e}")
            return image

    def process_images_batch(self, files):
        """批量处理图片 - 修复bug1: 返回正确格式给Gallery"""
        if not files:
            return []

        results = []
        for file in files:
            try:
                image = cv2.imread(file.name)
                if image is not None:
                    # 转换为RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    result = self.process_image(image_rgb)
                    if result is not None:
                        # Gallery需要PIL Image或文件路径，我们保存处理后的图片
                        temp_path = f"temp_result_{len(results)}.jpg"
                        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(temp_path, result_bgr)
                        results.append(temp_path)
            except Exception as e:
                print(f"❌ 处理文件失败 {file.name}: {e}")

        return results

    def process_video(self, video_file, progress=gr.Progress()):
        """处理视频文件 - 使用临时目录保存"""
        if video_file is None:
            return None, "请先上传视频文件"

        try:
            self.load_models_once()

            # 使用临时目录
            output_dir = Path("temp_output")
            output_dir.mkdir(exist_ok=True)

            input_path = video_file.name
            input_name = Path(input_path).stem
            output_path = output_dir / f"processed_{input_name}.mp4"

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return None, "无法打开视频文件"

            # 获取视频属性
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 使用H.264编码器
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 使用与摄像头相同的处理逻辑
                det_res = self.det_model.predict(frame, conf=STREAM_CONFIG["conf_thres"],
                                                 device=STREAM_CONFIG["device"], verbose=False)[0]
                boxes = det_res.boxes.xyxy.cpu().numpy() if det_res.boxes is not None else []

                faces = []
                coords = []
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue
                    faces.append(face)
                    coords.append((x1, y1, x2, y2))

                # 批量表情分类
                if faces:
                    try:
                        batch = []
                        for f in faces:
                            img = cv2.resize(f, (STREAM_CONFIG["img_size"], STREAM_CONFIG["img_size"]))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = img.transpose(2, 0, 1) / 255.0
                            batch.append(img)

                        batch_tensor = torch.from_numpy(np.stack(batch)).float()
                        batch_tensor = batch_tensor.to(STREAM_CONFIG["device"])

                        cls_res = self.cls_model.predict(batch_tensor, device=STREAM_CONFIG["device"], verbose=False)

                        # 使用与摄像头相同的类别权重
                        class_bias = torch.tensor([2.0, 1.7, 1.5, 0.9, 0.1, 7.0, 2.0], device=STREAM_CONFIG["device"])

                        # 绘制结果
                        for (x1, y1, x2, y2), res in zip(coords, cls_res):
                            if res.probs is None:
                                continue

                            logits = res.probs.data
                            if not isinstance(logits, torch.Tensor):
                                continue

                            # 加权并softmax
                            adjusted_logits = logits * class_bias
                            adjusted_probs = torch.nn.functional.softmax(adjusted_logits, dim=0)
                            idx = int(torch.argmax(adjusted_probs))
                            prob = float(adjusted_probs[idx])

                            label = f"{STREAM_CONFIG['names'][idx]} {prob:.2f}"
                            color = STREAM_CONFIG["colors"][idx % len(STREAM_CONFIG["colors"])]

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            self.draw_label(frame, label, (x1, y1), color)

                    except Exception as e:
                        print(f"⚠️ 分类失败: {e}")

                out.write(frame)

                frame_count += 1
                if frame_count % 10 == 0:  # 每10帧更新一次进度
                    progress_value = frame_count / total_frames
                    progress(progress_value, f"处理进度: {frame_count}/{total_frames} 帧")

            cap.release()
            out.release()

            return str(output_path), f"视频处理完成！共处理 {frame_count} 帧"

        except Exception as e:
            print(f"❌ 视频处理失败: {e}")
            return None, f"视频处理失败: {str(e)}"

    def process_camera_frame_optimized(self, frame):
        """🚀 优化版摄像头帧处理 - 使用跳帧和结果缓存"""
        try:
            self.frame_counter += 1

            # 决定是否需要重新检测
            should_detect = (
                    self.frame_counter % self.frame_skip == 0 or  # 跳帧间隔
                    self.last_detection_result is None or  # 没有缓存结果
                    self.cache_counter >= self.detection_cache_frames  # 缓存过期
            )

            if should_detect:
                # 执行完整的检测和分类
                detection_result = self._full_detection_and_classification(frame)
                self.last_detection_result = detection_result
                self.cache_counter = 0
                print(f"🔄 完整检测 - 帧 {self.frame_counter}")
            else:
                # 使用缓存结果，只更新坐标（可选）
                detection_result = self.last_detection_result
                self.cache_counter += 1
                print(f"⚡ 使用缓存 - 帧 {self.frame_counter}")

            # 绘制结果
            processed_frame = self._draw_detection_results(frame, detection_result)

            # 更新FPS统计
            self._update_fps_stats()

            return processed_frame

        except Exception as e:
            print(f"❌ 摄像头帧处理失败: {e}")
            return frame

    def _full_detection_and_classification(self, frame):
        """执行完整的检测和分类"""
        try:
            # 1. 人脸检测
            det_res = self.det_model.predict(
                frame,
                conf=STREAM_CONFIG["conf_thres"],
                device=STREAM_CONFIG["device"],
                verbose=False
            )[0]

            boxes = det_res.boxes.xyxy.cpu().numpy() if det_res.boxes is not None else []

            # 收集所有人脸区域
            faces = []
            coords = []
            valid_detections = []

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                faces.append(face)
                coords.append((x1, y1, x2, y2))

            # 2. 批量表情分类
            labels = []
            if faces:
                try:
                    # 批量准备所有图像
                    batch = []
                    for f in faces:
                        img = cv2.resize(f, (STREAM_CONFIG["img_size"], STREAM_CONFIG["img_size"]))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img.transpose(2, 0, 1) / 255.0
                        batch.append(img)

                    # 转换为张量并批量推理
                    batch_tensor = torch.from_numpy(np.stack(batch)).float()
                    batch_tensor = batch_tensor.to(STREAM_CONFIG["device"])

                    # 批量处理所有人脸
                    cls_res = self.cls_model.predict(
                        batch_tensor,
                        device=STREAM_CONFIG["device"],
                        verbose=False
                    )

                    # 类别权重
                    class_bias = torch.tensor(
                        [2.0, 1.7, 1.5, 0.8, 0.1, 7.0, 2.0],
                        device=STREAM_CONFIG["device"]
                    )

                    # 处理每个结果
                    for res in cls_res:
                        if res.probs is None:
                            labels.append("?")
                            continue

                        logits = res.probs.data
                        if not isinstance(logits, torch.Tensor):
                            labels.append("?")
                            continue

                        # 应用加权并softmax
                        adjusted_logits = logits * class_bias
                        adjusted_probs = torch.nn.functional.softmax(adjusted_logits, dim=0)
                        idx = int(torch.argmax(adjusted_probs))
                        prob = float(adjusted_probs[idx])

                        # 获取标签
                        if 0 <= idx < len(STREAM_CONFIG["names"]):
                            label = f"{STREAM_CONFIG['names'][idx]} {prob:.2f}"
                        else:
                            label = "unknown"
                        labels.append(label)

                except Exception as e:
                    print(f"⚠️ 批量分类失败: {e}")
                    labels = ["error"] * len(faces)

            # 返回检测结果
            return {
                'coords': coords,
                'labels': labels,
                'timestamp': time.time()
            }

        except Exception as e:
            print(f"❌ 完整检测失败: {e}")
            return {'coords': [], 'labels': [], 'timestamp': time.time()}

    def _draw_detection_results(self, frame, detection_result):
        """在帧上绘制检测结果"""
        if not detection_result or not detection_result['coords']:
            return frame

        try:
            coords = detection_result['coords']
            labels = detection_result['labels']

            # 绘制结果
            for (x1, y1, x2, y2), label in zip(coords, labels):
                try:
                    # 从标签中提取情感名称
                    emotion_name = label.split()[0] if ' ' in label else label

                    # 查找对应的颜色
                    if emotion_name in STREAM_CONFIG["names"]:
                        idx = STREAM_CONFIG["names"].index(emotion_name)
                    else:
                        idx = 0  # 默认为第一个颜色

                    color = STREAM_CONFIG["colors"][idx % len(STREAM_CONFIG["colors"])]

                    # 绘制检测框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # 绘制标签
                    self.draw_label(frame, label, (x1, y1), color)

                except Exception as e:
                    print(f"⚠️ 绘制失败: {e}")

            # 添加性能信息显示
            self._draw_performance_info(frame)

            return frame

        except Exception as e:
            print(f"❌ 绘制检测结果失败: {e}")
            return frame

    def _draw_performance_info(self, frame):
        """在帧上绘制性能信息"""
        try:
            # 性能信息
            perf_info = [
                f"FPS: {self.current_fps:.1f}",
                f"Skip: 1/{self.frame_skip}",
                f"Cache: {self.cache_counter}/{self.detection_cache_frames}"
            ]

            # 绘制半透明背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # 绘制文本
            for i, info in enumerate(perf_info):
                y = 30 + i * 20
                cv2.putText(frame, info, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"⚠️ 绘制性能信息失败: {e}")

    def _update_fps_stats(self):
        """更新FPS统计"""
        self.fps_counter += 1
        current_time = time.time()
        elapsed = current_time - self.fps_start_time

        if elapsed >= 1.0:  # 每秒更新一次FPS
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = current_time

    def get_available_cameras(self):
        """获取可用摄像头"""
        cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cameras.append(f"摄像头 {i}")
                cap.release()
        return cameras if cameras else ["无可用摄像头"]

    def start_camera(self, camera_choice, skip_frames=3):
        """启动摄像头 - 支持自定义跳帧设置"""
        if camera_choice == "无可用摄像头":
            return "没有可用的摄像头"

        # 停止现有摄像头
        if self.camera_active:
            self.stop_camera()
            time.sleep(0.5)

        try:
            # 解析摄像头ID
            camera_id = int(camera_choice.split()[-1])

            # 设置跳帧参数
            self.frame_skip = max(1, skip_frames)
            self.frame_counter = 0
            self.last_detection_result = None
            self.cache_counter = 0

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

                        # 🚀 使用优化版的帧处理
                        processed_frame = self.process_camera_frame_optimized(frame)

                        # 转换为RGB并更新当前帧
                        if processed_frame is not None:
                            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            self.current_frame = frame_rgb

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
        self.current_frame = None

        # 重置优化参数
        self.frame_counter = 0
        self.last_detection_result = None
        self.cache_counter = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()

        return "摄像头已停止"

    def get_camera_frame(self):
        """获取摄像头帧"""
        return self.current_frame if self.camera_active else None

    def update_skip_frames(self, skip_frames):
        """动态更新跳帧设置"""
        self.frame_skip = max(1, skip_frames)
        return f"跳帧设置已更新: 1/{self.frame_skip}"


# 创建应用实例
app = SimpleEmotionApp()


def create_interface():
    """创建简化的Gradio界面"""

    # 创建界面
    with gr.Blocks(title="情感识别系统", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎭 情感识别系统 (性能优化版)")
        gr.Markdown("基于 YOLO 的实时情感识别 - 支持跳帧优化")

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

                        # 添加操作提示
                        gr.Markdown("提示：使用鼠标滚轮或拖动滚动条查看所有图片")

                    with gr.Column():
                        # 修复bug1: 移除 rows 属性并增加高度
                        image_output = gr.Gallery(
                            label="识别结果",
                            columns=2,  # 保持2列布局
                            height=500,  # 设置固定高度
                            allow_preview=True,
                            show_download_button=True,
                            scroll_to_output=True  # 添加此属性确保结果可见
                        )

                image_btn.click(
                    fn=app.process_images_batch,
                    inputs=image_input,
                    outputs=image_output
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
                    outputs=[video_output, video_status]
                )

            # 🚀 实时摄像头 (优化版)
            with gr.Tab("📹 实时摄像头"):
                with gr.Row():
                    with gr.Column(scale=1):
                        camera_dropdown = gr.Dropdown(
                            choices=app.get_available_cameras(),
                            value=app.get_available_cameras()[0],
                            label="选择摄像头"
                        )

                        # 🚀 跳帧设置
                        skip_frames_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="跳帧设置 (1=无跳帧, 3=每3帧处理1帧)",
                            info="数值越大性能越好，但响应稍慢"
                        )

                        start_btn = gr.Button("启动摄像头", variant="primary")
                        stop_btn = gr.Button("停止摄像头", variant="secondary")

                        update_skip_btn = gr.Button("更新跳帧设置", variant="secondary")

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

                # 更新摄像头画面
                def update_camera():
                    frame = app.get_camera_frame()
                    return frame

                # 启动摄像头（带跳帧设置）
                def start_camera_with_skip(camera_choice, skip_frames):
                    return app.start_camera(camera_choice, skip_frames)

                # 启动摄像头
                start_btn.click(
                    fn=start_camera_with_skip,
                    inputs=[camera_dropdown, skip_frames_slider],
                    outputs=camera_status
                )

                # 停止摄像头
                stop_btn.click(
                    fn=app.stop_camera,
                    outputs=camera_status
                )

                # 动态更新跳帧设置
                update_skip_btn.click(
                    fn=app.update_skip_frames,
                    inputs=skip_frames_slider,
                    outputs=camera_status
                )

                # 创建定时器来更新摄像头画面
                camera_timer = gr.Timer(value=0.1)  # 每100ms更新一次
                camera_timer.tick(
                    fn=update_camera,
                    outputs=camera_output
                )

        # 使用说明
        with gr.Accordion("使用说明", open=False):
            gr.Markdown("""
            ### 🚀 性能优化功能
            - **跳帧处理**: 可设置每N帧处理一次，大幅提升性能
            - **结果缓存**: 非处理帧使用上次检测结果，保持流畅度
            - **实时FPS显示**: 界面显示当前帧率和优化状态
            - **动态调节**: 可在运行时调整跳帧设置

            ### 支持的情感类别
            - 😠 生气 (Angry)
            - 🤢 厌恶 (Disgust)
            - 😨 恐惧 (Fear)
            - 😊 快乐 (Happy)
            - 😢 悲伤 (Sad)
            - 😐 中性 (Neutral)
            - 😲 惊讶 (Surprise)

            ### 使用方法
            1. **图片识别**: 上传单张或多张图片进行批量处理
            2. **视频识别**: 上传视频文件，系统会逐帧处理并输出结果视频
            3. **实时摄像头**: 选择摄像头进行实时情感识别
               - 调节跳帧设置以平衡性能和响应速度
               - 查看实时FPS和优化状态

            ### 跳帧设置建议
            - **跳帧=1**: 无跳帧，最高精度，适合高性能设备
                - **跳帧=3**: 推荐设置，平衡性能和精度
                - **跳帧=5**: 高性能模式，适合低配置设备
                - **跳帧=10**: 极限性能模式，响应较慢但流畅

                ### 性能监控说明
                - **FPS**: 当前处理帧率
                - **Skip**: 当前跳帧设置
                - **Cache**: 缓存使用情况

                ### 注意事项
                - 确保图片中包含清晰的人脸
                - 视频处理可能需要较长时间，请耐心等待
                - 摄像头功能需要设备支持
                - 跳帧设置过高可能导致快速动作识别延迟

                ### 已优化的功能
                - ✅ 图片处理结果现在可以正常滚动查看
                - ✅ 视频处理完成后能正确显示结果
                - ✅ 摄像头画面现在能实时显示
                - 🚀 摄像头流现在支持跳帧优化，大幅提升性能
                - 🚀 添加了实时性能监控和动态参数调节
                """)

    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
