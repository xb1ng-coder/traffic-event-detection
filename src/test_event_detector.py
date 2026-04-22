"""快速测试交通事件检测模块"""
import cv2
from ultralytics import YOLO
from traffic_event_detector import TrafficEventDetector, EventType, EVENT_LABELS

model = YOLO('models/yolov8n.pt')
img = cv2.imread('../data/images/traffic1.jpg')
h, w = img.shape[:2]

detector = TrafficEventDetector(frame_width=w, frame_height=h, fps=30)
detector.set_normal_direction('right')

results = model(img, conf=0.3, verbose=False)
detections = []
if results[0].boxes is not None and len(results[0].boxes) > 0:
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        detections.append((cls_name, bbox, confidence))

print(f'图片尺寸: {w}x{h}')
print(f'YOLO检测到: {len(detections)} 个目标')
for d in detections:
    print(f'  - {d[0]}: conf={d[2]:.2f}, bbox={[round(x,1) for x in d[1]]}')

events = detector.detect(detections)
print(f'\n事件检测: {len(events)} 个事件')
for event in events:
    label = EVENT_LABELS.get(event.event_type, event.event_type.value)
    print(f'  [{event.severity}] {label}: {event.description}')

tracked = detector.get_tracked_objects_info()
print(f'\n追踪目标: {len(tracked)} 个')
for t in tracked:
    print(f'  #{t["track_id"]} {t["class_name"]} speed={t["avg_speed"]} frames={t["frame_count"]}')

print('\n测试完成!')
