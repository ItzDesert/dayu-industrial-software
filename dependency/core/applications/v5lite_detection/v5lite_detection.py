from pathlib import Path
from typing import List

import numpy as np

from core.lib.common import Context

from .v5lite_stream import yolov5_lite


class V5LiteDetector:
    """
    YOLOv5-Lite ONNX 检测器，封装为系统 Processor 可调用的接口。

    所有参数通过环境变量配置（见 template/processor/v5lite-detection.yaml）：
      V5LITE_MODEL_PATH     模型文件名（在挂载卷中解析）
      V5LITE_LABEL_PATH     类别名称文件名（在挂载卷中解析）
      V5LITE_CONF_THRESHOLD 置信度阈值
      V5LITE_NMS_THRESHOLD  NMS IoU 阈值
      V5LITE_MODEL_NAME     模型名称（可选）
    """

    def __init__(self):
        model_path = Context.get_file_path(Context.get_parameter('V5LITE_MODEL_PATH'))
        label_path = Context.get_file_path(Context.get_parameter('V5LITE_LABEL_PATH'))
        conf_threshold = float(Context.get_parameter('V5LITE_CONF_THRESHOLD', 0.2))
        nms_threshold = float(Context.get_parameter('V5LITE_NMS_THRESHOLD', 0.5))
        model_name = Context.get_parameter('V5LITE_MODEL_NAME', Path(model_path).stem)

        self._detector = yolov5_lite(
            model_path,
            label_path,
            confThreshold=conf_threshold,
            nmsThreshold=nms_threshold,
            model_name=model_name,
        )

    def infer(self, image: np.ndarray) -> dict:
        """对单帧图像运行检测，返回结构化结果字典。"""
        _, detections, perf_info = self._detector.detect(image)
        return {
            "detections": detections,
            "perf_info": perf_info,
        }

    def __call__(self, images: List[np.ndarray]) -> List[dict]:
        """对图像列表批量运行检测。"""
        return [self.infer(image) for image in images]
