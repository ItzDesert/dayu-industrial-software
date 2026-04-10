import cv2
import numpy as np
import onnxruntime as ort
import os
from pathlib import Path


class yolov5_lite:
    def __init__(self, model_pb_path, label_path, confThreshold=0.5, nmsThreshold=0.5, model_name=None):
        so = ort.SessionOptions()
        so.log_severity_level = 3

        self.model_name = model_name or Path(model_pb_path).stem
        self.net = ort.InferenceSession(model_pb_path, so)

        if os.path.exists(label_path):
            self.classes = list(map(lambda x: x.strip(), open(label_path, 'r').readlines()))
        else:
            self.classes = [f'class{i}' for i in range(80)]

        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.input_shape = (self.net.get_inputs()[0].shape[2], self.net.get_inputs()[0].shape[3])

    def letterBox(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_shape[1] - neww - left,
                                         cv2.BORDER_CONSTANT, value=0)
            else:
                newh, neww = int(self.input_shape[0] * hw_scale), self.input_shape[1]
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_shape[0] - newh - top, 0, 0,
                                         cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, self.input_shape, interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def postprocess(self, srcimg, outs, pad_hw):
        newh, neww, padh, padw = pad_hw
        frameHeight = srcimg.shape[0]
        frameWidth = srcimg.shape[1]
        ratioh, ratiow = frameHeight / newh, frameWidth / neww

        classIds = []
        confidences = []
        boxes = []

        for detection in outs:
            score = detection[4]
            classId = int(detection[5])
            if score > self.confThreshold:
                x1 = int((detection[0] - padw) * ratiow)
                y1 = int((detection[1] - padh) * ratioh)
                x2 = int((detection[2] - padw) * ratiow)
                y2 = int((detection[3] - padh) * ratioh)

                x1 = max(0, min(x1, frameWidth - 1))
                y1 = max(0, min(y1, frameHeight - 1))
                x2 = max(0, min(x2, frameWidth - 1))
                y2 = max(0, min(y2, frameHeight - 1))

                if x2 > x1 and y2 > y1:
                    classIds.append(classId)
                    confidences.append(float(score))
                    boxes.append([x1, y1, x2, y2])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)

        result_boxes = []
        result_scores = []
        result_classids = []

        for i in indices:
            result_boxes.append(boxes[i])
            result_scores.append(confidences[i])
            result_classids.append(classIds[i])

        result_boxes = np.array(result_boxes, dtype=int) if result_boxes else np.empty((0, 4), dtype=int)
        result_scores = np.array(result_scores, dtype=float) if result_scores else np.array([], dtype=float)
        result_classids = np.array(result_classids, dtype=int) if result_classids else np.array([], dtype=int)

        return result_boxes, result_scores, result_classids

    def detect(self, srcimg):
        img, newh, neww, top, left = self.letterBox(srcimg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0]

        result_boxes, result_scores, result_classids = self.postprocess(srcimg, outs, (newh, neww, top, left))
        return result_boxes, result_scores, result_classids
