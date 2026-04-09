import cv2
import time
import numpy as np
import argparse
import onnxruntime as ort
import os
import queue
import threading
import psutil
from datetime import datetime
from pathlib import Path


class NetworkStreamer:
    """网络流处理器"""
    
    def __init__(self, source, buffer_size=10):
        """
        初始化网络流
        
        Args:
            source: 视频源 (摄像头ID、RTSP URL等)
            buffer_size: 缓冲区大小
        """
        self.source = source
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.cap = None
        self.running = False
        self.thread = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
    def start_capture(self):
        """启动视频捕获线程"""
        if self.running:
            print("视频捕获已经在运行")
            return
        
        # 尝试打开视频源
        print(f"正在连接视频源: {self.source}")
        if self.source.isdigit():
            self.cap = cv2.VideoCapture(int(self.source))
        else:
            self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频源: {self.source}")
        
        # 获取视频信息
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"视频流信息: {self.width}x{self.height} @ {self.fps}fps")
        
        # 启动捕获线程
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        print("视频捕获已启动")
        
    def _capture_loop(self):
        """视频捕获循环"""
        reconnect_count = 0
        max_reconnects = 5
        reconnect_delay = 2
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("视频流中断，尝试重新连接...")
                    reconnect_count += 1
                    
                    if reconnect_count > max_reconnects:
                        print(f"重连失败次数过多 ({max_reconnects})，停止捕获")
                        self.running = False
                        break
                    
                    # 重新连接
                    time.sleep(reconnect_delay)
                    self.cap.release()
                    
                    if self.source.isdigit():
                        self.cap = cv2.VideoCapture(int(self.source))
                    else:
                        self.cap = cv2.VideoCapture(self.source)
                    
                    if not self.cap.isOpened():
                        print(f"重连失败: {self.source}")
                        continue
                    
                    print(f"重连成功: {self.source}")
                    reconnect_count = 0
                    continue
                
                # 重置重连计数
                reconnect_count = 0
                
                # 更新FPS计算
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # 如果队列已满，移除最旧的帧
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # 添加新帧到队列
                self.frame_queue.put(frame)
                
            except Exception as e:
                print(f"视频捕获错误: {e}")
                time.sleep(0.1)
        
        # 清理资源
        if self.cap is not None:
            self.cap.release()
    
    def get_frame(self):
        """获取最新的帧"""
        if not self.running:
            return None
        
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return frame
        except queue.Empty:
            return None
    
    def get_stream_fps(self):
        """获取实际流的FPS"""
        return self.fps
    
    def stop_capture(self):
        """停止视频捕获"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        
        if self.cap is not None:
            self.cap.release()
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        print("视频捕获已停止")


class yolov5_lite():
    def __init__(self, model_pb_path, label_path, confThreshold=0.5, nmsThreshold=0.5, model_name=None):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        
        # 记录模型名称
        self.model_name = model_name or Path(model_pb_path).stem
        
        print(f"加载模型: {model_pb_path}")
        self.net = ort.InferenceSession(model_pb_path, so)
        
        print(f"加载标签: {label_path}")
        if os.path.exists(label_path):
            self.classes = list(map(lambda x: x.strip(), open(label_path, 'r').readlines()))
            print(f"加载了 {len(self.classes)} 个类别")
        else:
            print(f"警告: 标签文件 {label_path} 不存在，使用默认类别")
            self.classes = [f"class{i}" for i in range(80)]

        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.input_shape = (self.net.get_inputs()[0].shape[2], self.net.get_inputs()[0].shape[3])
        print(f"模型输入尺寸: {self.input_shape}")
        
        # 性能统计
        self.frame_count = 0
        self.total_inference_time = 0
        self.total_preprocess_time = 0
        self.total_postprocess_time = 0
        self.fps_avg = 0
        self.detection_counts = []
        
        # 系统资源监控
        self.process = psutil.Process(os.getpid())

    def letterBox(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_shape[1] - neww - left, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.input_shape[0] * hw_scale), self.input_shape[1]
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_shape[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, self.input_shape, interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def postprocess(self, frame, outs, pad_hw):
        newh, neww, padh, padw = pad_hw
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / newh, frameWidth / neww
        
        # 存储检测结果
        classIds = []
        confidences = []
        boxes = []
        
        # 处理每个检测框
        for detection in outs:
            scores = detection[4]
            classId = detection[5]
            if scores > self.confThreshold:
                x1 = int((detection[0] - padw) * ratiow)
                y1 = int((detection[1] - padh) * ratioh)
                x2 = int((detection[2] - padw) * ratiow)
                y2 = int((detection[3] - padh) * ratioh)
                
                # 确保坐标在图像范围内
                x1 = max(0, min(x1, frameWidth - 1))
                y1 = max(0, min(y1, frameHeight - 1))
                x2 = max(0, min(x2, frameWidth - 1))
                y2 = max(0, min(y2, frameHeight - 1))
                
                # 只有当框有效时才添加
                if x2 > x1 and y2 > y1:
                    classIds.append(classId)
                    confidences.append(scores)
                    boxes.append([x1, y1, x2, y2])

        # 应用非极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        
        # 检测结果
        results = []
        
        # 在图像上绘制检测框
        for i in indices:
            box = boxes[i]
            classId = int(classIds[i])
            confidence = float(confidences[i])
            
            # 获取类别名称
            if classId < len(self.classes):
                class_name = self.classes[classId]
            else:
                class_name = f"class{classId}"
                
            results.append({
                "class": class_name,
                "confidence": confidence,
                "box": box
            })
            
            # 绘制框和标签
            frame = self.drawPred(frame, classId, confidence, box[0], box[1], box[2], box[3])
            
        return frame, results

    def drawPred(self, frame, classId, conf, x1, y1, x2, y2):
        # 为不同类别使用不同颜色
        colors = [
            (0, 255, 0), (0, 0, 255), (255, 0, 0), 
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128)
        ]
        color = colors[classId % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        # 准备标签文本
        if classId < len(self.classes):
            class_name = self.classes[classId]
        else:
            class_name = f"class{classId}"
            
        label = f'{class_name}: {conf:.2f}'

        # 获取文本大小
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # 确保标签位置在图像内
        y1_label = max(y1, labelSize[1])
        
        # 绘制标签背景
        cv2.rectangle(frame, 
                     (x1, y1_label - labelSize[1] - 10), 
                     (x1 + labelSize[0], y1_label), 
                     (255, 255, 255), 
                     cv2.FILLED)
        
        # 绘制标签文本
        cv2.putText(frame, label, (x1, y1_label - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
        
        return frame

    def get_cpu_usage(self):
        """获取当前进程的CPU使用率"""
        return self.process.cpu_percent()
    
    def get_memory_usage(self):
        """获取当前进程的内存使用率"""
        return self.process.memory_percent()

    def detect(self, srcimg):
        # 总时间计时开始
        t_start = time.time()
        
        # 图像预处理计时
        t_preprocess_start = time.time()
        img, newh, neww, top, left = self.letterBox(srcimg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        preprocess_time = time.time() - t_preprocess_start
        
        # 模型推理计时
        t_inference_start = time.time()
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0]
        inference_time = time.time() - t_inference_start
        
        # 后处理计时
        t_postprocess_start = time.time()
        srcimg, results = self.postprocess(srcimg, outs, (newh, neww, top, left))
        postprocess_time = time.time() - t_postprocess_start
        
        # 总时间计算
        total_time = time.time() - t_start
        
        # 更新性能统计
        self.frame_count += 1
        self.total_inference_time += inference_time
        self.total_preprocess_time += preprocess_time
        self.total_postprocess_time += postprocess_time
        self.fps_avg = self.frame_count / (self.total_inference_time + self.total_preprocess_time + self.total_postprocess_time)
        self.detection_counts.append(len(results))
        
        # 添加推理时间信息
        infer_time = f'Inference: {int(inference_time * 1000)}ms | FPS: {self.fps_avg:.1f}'
        cv2.putText(srcimg, infer_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), thickness=2)
        cv2.putText(srcimg, infer_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=1)
        
        # 添加检测计数
        if len(self.detection_counts) > 30:  # 计算最近30帧的平均检测数
            avg_detections = sum(self.detection_counts[-30:]) / 30
        else:
            avg_detections = sum(self.detection_counts) / len(self.detection_counts)
            
        det_info = f'Detections: {len(results)} | Avg: {avg_detections:.1f}'
        cv2.putText(srcimg, det_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), thickness=2)
        cv2.putText(srcimg, det_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=1)
        
        # 添加模型信息
        model_info = f'Model: {self.model_name} | Size: {self.input_shape[0]}x{self.input_shape[1]}'
        cv2.putText(srcimg, model_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), thickness=2)
        cv2.putText(srcimg, model_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=1)
        
        # 添加时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(srcimg, timestamp, (10, srcimg.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)
        cv2.putText(srcimg, timestamp, (10, srcimg.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=1)
        
        # 构建性能信息字典
        performance_info = {
            'total_time': total_time,
            'preprocess_time': preprocess_time,
            'inference_time': inference_time,
            'postprocess_time': postprocess_time,
            'frame_count': self.frame_count,
            'fps': 1.0 / total_time if total_time > 0 else 0,
            'avg_fps': self.fps_avg,
            'model_name': self.model_name,
            'input_shape': self.input_shape
        }
        
        return srcimg, results, performance_info


def process_stream(model_path, stream_source, output_folder, class_file, conf_threshold=0.45, nms_threshold=0.5, 
                  save_interval=0, record_video=False, show=True, fps_limit=30, model_name=None):
    """
    处理视频流
    
    Args:
        model_path: 模型路径
        stream_source: 视频流源 (摄像头ID、RTSP URL等)
        output_folder: 输出文件夹
        class_file: 类别文件路径
        conf_threshold: 置信度阈值
        nms_threshold: NMS阈值
        save_interval: 保存图像的间隔(秒)，0表示不保存
        record_video: 是否录制视频
        show: 是否显示视频
        fps_limit: FPS限制
        model_name: 模型名称（可选）
    """
    # 创建输出文件夹
    if save_interval > 0 or record_video:
        os.makedirs(output_folder, exist_ok=True)
    
    # 创建检测器
    detector = yolov5_lite(model_path, class_file, confThreshold=conf_threshold, nmsThreshold=nms_threshold, model_name=model_name)
    
    # 创建视频流
    streamer = NetworkStreamer(stream_source)
    streamer.start_capture()
    
    # 视频录制设置
    video_writer = None
    if record_video:
        # 等待获取第一帧以确定视频尺寸
        first_frame = None
        while first_frame is None:
            first_frame = streamer.get_frame()
            if first_frame is None:
                print("等待视频流...")
                time.sleep(0.5)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = os.path.join(output_folder, f"stream_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, 
                                      (first_frame.shape[1], first_frame.shape[0]))
        print(f"视频录制已开始: {video_path}")
    
    # 创建性能日志文件
    if save_interval > 0 or record_video:
        perf_log_path = os.path.join(output_folder, f"performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(perf_log_path, 'w') as f:
            f.write("timestamp,frame,model,total_time_ms,preprocess_time_ms,inference_time_ms,postprocess_time_ms,fps,detections\n")
    
    # 处理循环
    last_save_time = time.time()
    last_frame_time = time.time()
    frame_interval = 1.0 / fps_limit
    
    try:
        print("\n" + "=" * 80)
        print("开始处理视频流... 按 'q' 退出，按 's' 保存当前帧")
        print("=" * 80)
        print(f"{'时间戳':^19} | {'帧':^5} | {'模型':^10} | {'总时间':^8} | {'预处理':^8} | {'推理':^8} | {'后处理':^8} | {'FPS':^6} | {'检测数':^6} | {'检测结果'}")
        print("-" * 80)
        
        while True:
            # FPS限制
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)  # 短暂休眠以减少CPU使用
                continue
            
            last_frame_time = current_time
            
            # 获取帧
            frame = streamer.get_frame()
            if frame is None:
                print("等待视频流...")
                time.sleep(0.5)
                continue
            
            # 执行检测
            result_frame, detections, perf_info = detector.detect(frame)
            
            # 获取CPU和内存使用情况
            cpu_usage = detector.get_cpu_usage()
            memory_usage = detector.get_memory_usage()
            
            # 在控制台输出详细信息
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 基础性能信息
            print(f"{timestamp} | {perf_info['frame_count']:5d} | {perf_info['model_name']:10s} | "
                  f"{perf_info['total_time']*1000:8.1f} | {perf_info['preprocess_time']*1000:8.1f} | "
                  f"{perf_info['inference_time']*1000:8.1f} | {perf_info['postprocess_time']*1000:8.1f} | "
                  f"{perf_info['fps']:6.1f} | {len(detections):6d} | ", end="")
            
            # 检测结果详情
            if len(detections) > 0:
                detection_list = ', '.join([f"{d['class']}({d['confidence']:.2f})" for d in detections])
                print(f"{detection_list}")
            else:
                print("无检测结果")
            
            # 记录性能日志
            if (save_interval > 0 or record_video) and os.path.exists(output_folder):
                with open(perf_log_path, 'a') as f:
                    f.write(f"{timestamp},{perf_info['frame_count']},{perf_info['model_name']},"
                            f"{perf_info['total_time']*1000:.1f},{perf_info['preprocess_time']*1000:.1f},"
                            f"{perf_info['inference_time']*1000:.1f},{perf_info['postprocess_time']*1000:.1f},"
                            f"{perf_info['fps']:.1f},{len(detections)}\n")
            
            # 保存间隔帧
            if save_interval > 0 and (current_time - last_save_time) >= save_interval:
                save_path = os.path.join(output_folder, f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(save_path, result_frame)
                print(f"已保存帧: {save_path}")
                last_save_time = current_time
            
            # 录制视频
            if video_writer is not None:
                video_writer.write(result_frame)
            
            # 显示结果
            if show:
                # 添加系统资源信息
                resource_info = f'CPU: {cpu_usage:.1f}% | Memory: {memory_usage:.1f}%'
                cv2.putText(result_frame, resource_info, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), thickness=2)
                cv2.putText(result_frame, resource_info, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), thickness=1)
                
                cv2.namedWindow("YOLOv5-Lite Stream Detection", cv2.WINDOW_NORMAL)
                cv2.imshow("YOLOv5-Lite Stream Detection", result_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户退出")
                    break
                elif key == ord('s'):
                    # 手动保存当前帧
                    save_path = os.path.join(output_folder, f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(save_path, result_frame)
                    print(f"已手动保存帧: {save_path}")
    
    except KeyboardInterrupt:
        print("\n程序被中断")
    
    finally:
        # 清理资源
        streamer.stop_capture()
        
        if video_writer is not None:
            video_writer.release()
            print("视频录制已结束")
        
        if show:
            cv2.destroyAllWindows()
        
        # 打印统计信息
        if detector.frame_count > 0:
            print("\n" + "=" * 80)
            print("性能统计:")
            print(f"处理帧数: {detector.frame_count}")
            print(f"平均FPS: {detector.fps_avg:.2f}")
            print(f"平均预处理时间: {(detector.total_preprocess_time / detector.frame_count) * 1000:.2f}ms")
            print(f"平均推理时间: {(detector.total_inference_time / detector.frame_count) * 1000:.2f}ms")
            print(f"平均后处理时间: {(detector.total_postprocess_time / detector.frame_count) * 1000:.2f}ms")
            
            if len(detector.detection_counts) > 0:
                avg_detections = sum(detector.detection_counts) / len(detector.detection_counts)
                print(f"平均每帧检测对象数: {avg_detections:.2f}")
            
            print("=" * 80)
        
        print("处理完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv5-Lite 视频流检测")
    parser.add_argument('--source', type=str, default='rtsp://192.168.3.8:8554/stream0', 
                        help="视频源 (0=摄像头, rtsp://... =RTSP流, http://... =HTTP流)")
    parser.add_argument('--modelpath', type=str, default='./best.onnx', 
                        help="ONNX模型文件路径")
    parser.add_argument('--classfile', type=str, default='./label_names', 
                        help="类别名称文件路径")
    parser.add_argument('--outfolder', type=str, default='stream_results', 
                        help="输出文件夹路径")
    parser.add_argument('--confThreshold', default=0.45, type=float, 
                        help='置信度阈值')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, 
                        help='NMS IoU阈值')
    parser.add_argument('--save_interval', default=0, type=int, 
                        help='保存图像的间隔(秒)，0表示不保存')
    parser.add_argument('--record', action='store_true', 
                        help='录制视频')
    parser.add_argument('--no_show', action='store_true', 
                        help='不显示视频窗口')
    parser.add_argument('--fps_limit', default=30, type=int, 
                        help='FPS限制')
    parser.add_argument('--model_name', type=str, default=None,
                        help='模型名称（可选，默认使用文件名）')
    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.modelpath):
        print(f"错误: 模型文件不存在: {args.modelpath}")
        exit(1)
    
    # 如果未指定模型名称，使用文件名
    if args.model_name is None:
        args.model_name = Path(args.modelpath).stem
    
    # 打印参数信息
    print("=" * 50)
    print("YOLOv5-Lite 视频流检测")
    print("=" * 50)
    print(f"视频源: {args.source}")
    print(f"模型路径: {args.modelpath}")
    print(f"类别文件: {args.classfile}")
    print(f"输出文件夹: {args.outfolder}")
    print(f"置信度阈值: {args.confThreshold}")
    print(f"NMS阈值: {args.nmsThreshold}")
    print(f"保存间隔: {args.save_interval}秒")
    print(f"录制视频: {'是' if args.record else '否'}")
    print(f"显示视频: {'否' if args.no_show else '是'}")
    print(f"FPS限制: {args.fps_limit}")
    print(f"模型名称: {args.model_name}")
    print("=" * 50)
    
    # 处理视频流
    process_stream(
        args.modelpath,
        args.source,
        args.outfolder,
        args.classfile,
        args.confThreshold,
        args.nmsThreshold,
        args.save_interval,
        args.record,
        not args.no_show,
        args.fps_limit,
        args.model_name
    )
