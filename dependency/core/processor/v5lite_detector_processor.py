import cv2

from .processor import Processor

from core.lib.estimation import Timer
from core.lib.content import Task
from core.lib.common import LOGGER, Context
from core.lib.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.PROCESSOR, alias='v5lite_detector_processor')
class V5LiteDetectorProcessor(Processor):
    """
    基于 YOLOv5-Lite ONNX 的工业缺陷检测 Processor。

    从 Task 携带的视频文件中逐帧读取，调用 V5LiteDetector 完成推理，
    并将每帧的检测结果列表写入 Task 内容。

    依赖 SERVICE_NAME=processor-v5lite-detection 加载对应 Application 模块，
    或通过 DETECTOR_PARAMETERS 传入构造参数。
    """

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

        if not image_list:
            LOGGER.critical('[V5LiteDetectorProcessor] No frames decoded from file')
            LOGGER.critical(f'Source: {task.get_source_id()}, Task: {task.get_task_id()}')
            LOGGER.critical(f'file_path: {task.get_file_path()}')
            return None

        result = self.infer(image_list)
        task = self.get_scenario(result, task)
        task.set_current_content(result)
        return task

    def infer(self, images):
        assert self.detector, 'No V5LiteDetector defined!'
        with Timer(f'V5Lite Detection / {len(images)} frame(s)'):
            return self.detector(images)
