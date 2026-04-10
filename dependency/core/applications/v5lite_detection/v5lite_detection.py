from pathlib import Path
from typing import List

import numpy as np

from core.lib.common import Context

from .v5lite_stream import yolov5_lite


class V5LiteDetector:

    def __init__(self, model_path=None, label_path=None, conf_threshold=None, nms_threshold=None, model_name=None):
        # 兼容通过 DETECTOR_PARAMETERS 传入 kwargs 和通过旧 V5LITE_* 独立环境变量两种方式
        if model_path is None:
            model_path = Context.get_parameter('V5LITE_MODEL_PATH')
        if label_path is None:
            label_path = Context.get_parameter('V5LITE_LABEL_PATH')
        if conf_threshold is None:
            conf_threshold = Context.get_parameter('V5LITE_CONF_THRESHOLD', 0.2)
        if nms_threshold is None:
            nms_threshold = Context.get_parameter('V5LITE_NMS_THRESHOLD', 0.5)
        if model_name is None:
            model_name = Context.get_parameter('V5LITE_MODEL_NAME', None)

        model_path = Context.get_file_path(model_path)
        label_path = Context.get_file_path(label_path)

        if model_name is None:
            model_name = Path(model_path).stem

        self.model = yolov5_lite(
            model_path,
            label_path,
            confThreshold=float(conf_threshold),
            nmsThreshold=float(nms_threshold),
            model_name=model_name,
        )

    def infer(self, image: np.ndarray):
        result_boxes, result_scores, result_classids = self.model.detect(image)
        return result_boxes, result_scores, result_classids

    def __call__(self, images: List[np.ndarray]):
        output = []
        for image in images:
            result_boxes, result_scores, result_classids = self.infer(image)
            output.append((result_boxes, result_scores, result_classids))
        return output
