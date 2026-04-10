from pathlib import Path
from typing import List

import numpy as np

from core.lib.common import Context

from .v5lite_stream import yolov5_lite


class V5LiteDetector:

    def __init__(self, model_path, label_path, conf_threshold=0.2, nms_threshold=0.5, model_name=None):
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
