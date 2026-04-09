# grpc_server.py
# gRPC 服务端，接收并处理视频帧，通过 Redis 发布结果
import time
import base64
import redis
import cv2
import json
from v5lite_stream import yolov5_lite

def get_param():
    params = {
        "small_modelpath": "/home/zs/Project_completion_1222/models/v5lite_s.onnx",
        "large_modelpath": "/home/zs/Project_completion_1222/models/v5lite_g.onnx",
        "small_model_name": "v5lite_s",
        "large_model_name": "v5lite_g",
        "classfile": "/home/zs/Project_completion_1222/models/label_names",
        "confThreshold": 0.2,
        "nmsThreshold": 0.5,
    }

    return params

# 初始化检测
detect_config = get_param()
detector = yolov5_lite(detect_config["small_modelpath"],
                        detect_config["classfile"],
                        confThreshold=detect_config["confThreshold"],
                        nmsThreshold=detect_config["nmsThreshold"],
                        model_name=detect_config["small_model_name"]
                        )
camera_id = 0
cap = cv2.VideoCapture(camera_id)
redis_server = redis.Redis(host="localhost", port=6379, decode_responses=True)

window_name = "Detection Result"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(camera_id)
        continue
    result_frame, detections, perf_info = detector.detect(frame)

    _, jpeg = cv2.imencode('.jpg', result_frame)
    base64_img = base64.b64encode(jpeg).decode()
    # 显示窗口
    cv2.imshow(window_name, result_frame)

    # 按 q 或 ESC 退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

    # Redis publish
    redis_server.publish("video_channel_1", json.dumps({
        "camera_id": camera_id,
        "image": base64_img,
        "detections": detections,
        "perf_info": perf_info,
        "create_timestamp": time.time()
    }))
    time.sleep(0.1)

cap.release()
cv2.destroyAllWindows()
    # Redis publish
    # redis_server.publish("video_channel_1", json.dumps({
    #     "camera_id": camera_id,
    #     # "image": base64_img,
    #     "detections": detections,
    #     "perf_info": perf_info,
    #     "create_timestamp": time.time()
    # }))

    


