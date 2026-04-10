import abc

from core.lib.common import ClassFactory, ClassType, EncodeOps, LOGGER
from core.lib.content import Task

from .image_visualizer import ImageVisualizer

__all__ = ('ROIFrameVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='roi_frame')
class ROIFrameVisualizer(ImageVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.roi_service = kwargs.get('roi_service', None)
        # 可选：通过 hook_params 传入类名列表，如 label_names: ['cat', 'dog', ...]
        self.label_names = kwargs.get('label_names', None)

    def _build_labels(self, classids, scores):
        labels = []
        for cls_id, score in zip(classids, scores):
            cls_id = int(cls_id)
            if self.label_names and 0 <= cls_id < len(self.label_names):
                name = self.label_names[cls_id]
            else:
                name = f'cls{cls_id}'
            labels.append(f'{name}: {float(score):.2f}')
        return labels

    def __call__(self, task: Task):
        try:
            if self.roi_service:
                content = task.get_dag().get_node(self.roi_service).service.get_content_data()
            else:
                content = task.get_first_content()
        except Exception:
            content = task.get_first_content()
        file_path = task.get_file_path()

        try:
            image = self.get_first_frame_from_video(file_path)
            frame_result = content[0]

            # 当内容包含 (boxes, scores, classids) 三元组时，绘制带标签的 bbox
            if isinstance(frame_result, (list, tuple)) and len(frame_result) >= 3:
                bboxes = list(frame_result[0])
                scores = list(frame_result[1])
                classids = list(frame_result[2])
                labels = self._build_labels(classids, scores)
                image = self.draw_bboxes_and_labels(image, bboxes, labels)
            else:
                image = self.draw_bboxes(image, list(frame_result[0]))

            base64_data = EncodeOps.encode_image(image)
        except Exception as e:
            import cv2
            base64_data = EncodeOps.encode_image(
                cv2.imread(self.default_visualization_image)
            )
            LOGGER.warning(f'Video visualization fetch failed: {str(e)}')
            LOGGER.exception(e)

        return {self.variables[0]: base64_data}
