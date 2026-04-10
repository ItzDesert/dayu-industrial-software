from typing import List

import cv2
import numpy as np

from .processor import Processor

from core.lib.estimation import Timer
from core.lib.content import Task
from core.lib.common import LOGGER, Context, convert_ndarray_to_list
from core.lib.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.PROCESSOR, alias='v5lite_detector_processor')
class V5LiteDetectorProcessor(Processor):

    def __init__(self):
        super().__init__()
        self.detector = Context.get_instance('Detector')

    def __call__(self, task: Task):
        data_file_path = task.get_file_path()
        cap = cv2.VideoCapture(data_file_path)
        image_list = []
        success, frame = cap.read()
        while success:
            image_list.append(frame)
            success, frame = cap.read()
        cap.release()

        if len(image_list) == 0:
            LOGGER.critical('[V5LiteDetectorProcessor] No frames decoded from file')
            LOGGER.critical(f'Source: {task.get_source_id()}, Task: {task.get_task_id()}')
            LOGGER.critical(f'file_path: {task.get_file_path()}')
            return None

        result = self.infer(image_list)
        task = self.get_scenario(result, task)
        task.set_current_content(convert_ndarray_to_list(result))

        return task

    def infer(self, images: List[np.ndarray]):
        assert self.detector, 'No Detector defined!'
        with Timer(f'V5Lite Detection / {len(images)} frame(s)'):
            return self.detector(images)
