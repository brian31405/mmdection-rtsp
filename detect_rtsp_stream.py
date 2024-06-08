import cv2
from mmdet.apis import init_detector, inference_detector
import mmcv

# 配置文件和模型權重文件
config_file = 'faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# 初始化檢測模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# RTSP串流的URL
rtsp_url = 'rtsp://admin:123456@192.168.0.126:554/chID=4&streamType=main'

# 打開RTSP串流
cap = cv2.VideoCapture(rtsp_url)

# 檢查是否成功打開RTSP串流
if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

# 定義要檢測的類別
classes_to_detect = ['person', 'bicycle', 'car', 'motorcycle']

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame.")
        break
    
    # 進行目標檢測
    result = inference_detector(model, frame)
    
    # 繪製檢測結果
    for i, class_name in enumerate(model.CLASSES):
        if class_name in classes_to_detect:
            bboxes = result[i]
            for bbox in bboxes:
                if bbox[4] > 0.60:  # 置信度閾值
                    x1, y1, x2, y2, score = bbox
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label_text = f'{class_name}: {score:.2f}'
                    cv2.putText(frame, label_text, (int(x1), int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 顯示結果
    cv2.imshow('RTSP Stream', frame)
    
    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
