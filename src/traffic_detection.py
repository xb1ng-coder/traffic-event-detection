"""
交通事件实时检测系统
基于YOLOv8目标检测 + 时空分析 + 规则引擎

支持检测事件:
  - 车辆停止/违停
  - 车辆逆行
  - 行人闯入车道
  - 交通拥堵
  - 碰撞/事故
  - 车辆超速
  - 车辆缓行

使用方法:
  摄像头模式:   python src/traffic_detection.py --source camera
  视频文件模式: python src/traffic_detection.py --source video --input data/videos/test.mp4
  图片目录模式: python src/traffic_detection.py --source images --input data/images/
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import argparse
import glob

from traffic_event_detector import (
    TrafficEventDetector, EventType, SEVERITY_LEVELS, EVENT_LABELS
)


# 颜色配置 (BGR)
COLORS = {
    'car': (255, 100, 100),
    'truck': (100, 255, 100),
    'bus': (100, 100, 255),
    'motorcycle': (255, 255, 100),
    'bicycle': (255, 100, 255),
    'person': (100, 255, 255),
}

SEVERITY_COLORS = {
    'danger': (0, 0, 255),    # 红色
    'warning': (0, 165, 255), # 橙色
    'info': (0, 255, 255),    # 黄色
}


class TrafficDetectionSystem:
    """交通事件检测系统"""

    def __init__(self, model_path='models/yolov8n.pt', source='camera',
                 input_path=None, confidence=0.3, direction='right'):
        self.model_path = model_path
        self.source = source
        self.input_path = input_path
        self.confidence = confidence
        self.direction = direction

        # 初始化组件
        self.model = None
        self.cap = None
        self.event_detector = None
        self.is_running = False

        # 统计
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0
        self.total_events = defaultdict(int)

    def load_model(self):
        """加载YOLO模型"""
        print("\n[1/3] 加载YOLO模型...")
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"  从本地加载: {self.model_path}")
            else:
                print("  下载YOLOv8n模型...")
                self.model = YOLO('yolov8n.pt')
                os.makedirs('models', exist_ok=True)
                self.model.save(self.model_path)
                print(f"  模型已保存: {self.model_path}")
        except Exception as e:
            print(f"  模型加载失败: {e}")
            return False

        # 预热
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)
        print("  模型就绪")
        return True

    def init_source(self):
        """初始化视频源"""
        print("\n[2/3] 初始化视频源...")

        if self.source == 'camera':
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("  无法打开摄像头")
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            print(f"  摄像头: {width}x{height} @ {fps:.0f}FPS")

        elif self.source == 'video':
            if not self.input_path or not os.path.exists(self.input_path):
                print(f"  视频文件不存在: {self.input_path}")
                return False
            self.cap = cv2.VideoCapture(self.input_path)
            if not self.cap.isOpened():
                print(f"  无法打开视频: {self.input_path}")
                return False
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            print(f"  视频: {width}x{height} @ {fps:.0f}FPS")

        elif self.source == 'images':
            if not self.input_path or not os.path.isdir(self.input_path):
                print(f"  图片目录不存在: {self.input_path}")
                return False
            width, height, fps = 1280, 720, 30
            print(f"  图片目录: {self.input_path}")
        else:
            print(f"  不支持的源类型: {self.source}")
            return False

        # 初始化事件检测器
        self.event_detector = TrafficEventDetector(
            frame_width=width, frame_height=height, fps=fps
        )
        self.event_detector.set_normal_direction(self.direction)

        return True

    def _parse_yolo_results(self, results):
        """解析YOLO检测结果为统一格式"""
        detections = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                detections.append((cls_name, bbox, confidence))
        return detections

    def _draw_tracks(self, frame, tracks):
        """绘制追踪信息"""
        for tid, track in tracks.items():
            x1, y1, x2, y2 = map(int, track.bbox)
            color = COLORS.get(track.class_name, (200, 200, 200))

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 绘制轨迹
            if len(track.positions) >= 2:
                pts = np.array(track.positions[-30:], dtype=np.int32)
                cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)

            # 绘制速度和ID标签
            avg_speed = np.mean(track.speeds[-5:]) if track.speeds else 0
            label = f"#{tid} {track.class_name} v={avg_speed:.1f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 停止标记
            if track.stopped_frames >= self.event_detector.config['stop_min_frames'] // 2:
                cx, cy = int(track.center[0]), int(track.center[1])
                cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 2)
                cv2.putText(frame, "STOP", (cx - 18, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

    def _draw_events_panel(self, frame, events):
        """绘制事件面板"""
        if not events:
            return frame

        h, w = frame.shape[:2]
        panel_w = 360
        panel_h = 40 + len(events) * 35
        panel_x = w - panel_w - 10
        panel_y = 10

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # 标题
        cv2.putText(frame, "TRAFFIC EVENTS", (panel_x + 10, panel_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 事件列表
        y_off = 50
        for event in events:
            color = SEVERITY_COLORS.get(event.severity, (255, 255, 255))
            label = EVENT_LABELS.get(event.event_type, event.event_type.value)
            text = f"[{event.severity.upper()}] {label}"

            # 绘制严重等级指示条
            cv2.rectangle(frame, (panel_x + 5, panel_y + y_off - 12),
                          (panel_x + 9, panel_y + y_off + 8), color, -1)

            cv2.putText(frame, text, (panel_x + 14, panel_y + y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

            # 事件位置标记
            loc_x = int(event.location[0] * w)
            loc_y = int(event.location[1] * h)
            cv2.drawMarker(frame, (loc_x, loc_y), color,
                           cv2.MARKER_CROSS, 30, 2)

            y_off += 35

        return frame

    def _draw_stats_panel(self, frame):
        """绘制统计信息面板"""
        h, w = frame.shape[:2]
        panel_w = 280
        panel_h = 180
        panel_x, panel_y = 10, 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        y = panel_y + 25
        cv2.putText(frame, "Traffic Detection System", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        y += 22
        cv2.putText(frame, f"Frame: {self.frame_count}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 22
        cv2.putText(frame, f"Direction: {self.direction}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 25

        # 追踪目标数
        tracked = len(self.event_detector.tracker.tracks) if self.event_detector else 0
        cv2.putText(frame, f"Tracked Objects: {tracked}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 1)
        y += 22

        # 事件统计
        summary = self.event_detector.get_event_summary() if self.event_detector else {}
        total = sum(summary.values())
        cv2.putText(frame, f"Total Events: {total}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        return frame

    def _draw_road_region(self, frame):
        """绘制道路区域"""
        if not self.event_detector:
            return frame
        r = self.event_detector.road_region
        x1 = int(r[0] * self.event_detector.frame_width)
        y1 = int(r[1] * self.event_detector.frame_height)
        x2 = int(r[2] * self.event_detector.frame_width)
        y2 = int(r[3] * self.event_detector.frame_height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
        cv2.putText(frame, "Road Region", (x1 + 5, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        return frame

    def _process_frame(self, frame):
        """处理单帧"""
        self.frame_count += 1

        # YOLO检测
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = self._parse_yolo_results(results)

        # 交通事件检测
        events = self.event_detector.detect(detections)

        # 更新事件统计
        for event in events:
            self.total_events[event.event_type.value] += 1

        # 绘制
        tracks = self.event_detector.tracker.tracks
        frame = self._draw_tracks(frame, tracks)
        frame = self._draw_road_region(frame)
        frame = self._draw_events_panel(frame, events)
        frame = self._draw_stats_panel(frame)

        # 计算FPS
        current_time = time.time()
        if self.start_time > 0:
            elapsed = current_time - self.start_time
            if elapsed > 0:
                self.fps = 1.0 / elapsed
        self.start_time = current_time

        return frame, events

    def run_camera_or_video(self):
        """运行摄像头/视频检测"""
        if not self.cap or not self.model:
            print("初始化失败")
            return

        self.is_running = True
        paused = False

        print("\n[3/3] 开始检测")
        print("  按键说明:")
        print("    q - 退出")
        print("    s - 保存截图")
        print("    p - 暂停/继续")
        print("    r - 重新设置道路区域")
        print("    +/- - 调整检测灵敏度")
        print()

        while self.is_running:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    # 视频结束则循环播放
                    if self.source == 'video':
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print("无法读取帧")
                    break

                frame, events = self._process_frame(frame)

                # 事件控制台输出
                for event in events:
                    label = EVENT_LABELS.get(event.event_type, event.event_type.value)
                    print(f"  [{event.severity.upper()}] {label}: {event.description}")

                cv2.imshow('Traffic Event Detection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = f"results/capture_{ts}.jpg"
                os.makedirs("results", exist_ok=True)
                cv2.imwrite(path, frame)
                print(f"  截图已保存: {path}")
            elif key == ord('p'):
                paused = not paused
                print(f"  {'暂停' if paused else '继续'}")
            elif key == ord('+') or key == ord('='):
                # 提高灵敏度（降低置信度阈值）
                self.confidence = max(0.1, self.confidence - 0.05)
                print(f"  灵敏度提高, 置信度阈值: {self.confidence:.2f}")
            elif key == ord('-'):
                # 降低灵敏度
                self.confidence = min(0.9, self.confidence + 0.05)
                print(f"  灵敏度降低, 置信度阈值: {self.confidence:.2f}")

        self._cleanup()

    def run_images(self):
        """运行图片目录检测"""
        if not self.model or not self.input_path:
            print("初始化失败")
            return

        formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        image_paths = []
        for fmt in formats:
            image_paths.extend(glob.glob(os.path.join(self.input_path, fmt)))
        image_paths = sorted(image_paths)

        if not image_paths:
            print(f"  未找到图片: {self.input_path}")
            return

        print(f"\n[3/3] 处理 {len(image_paths)} 张图片")

        os.makedirs("results", exist_ok=True)

        for i, img_path in enumerate(image_paths, 1):
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            # 重置追踪器（每张图片独立检测）
            self.event_detector.tracker = type(self.event_detector.tracker)(
                max_disappeared=self.event_detector.tracker.max_disappeared,
                iou_threshold=self.event_detector.tracker.iou_threshold
            )
            self.event_detector.frame_idx = 0

            # 调整检测器帧尺寸
            h, w = frame.shape[:2]
            self.event_detector.frame_width = w
            self.event_detector.frame_height = h

            frame, events = self._process_frame(frame)

            # 保存结果
            filename = os.path.splitext(os.path.basename(img_path))[0]
            output_path = f"results/{filename}_event_detected.jpg"
            cv2.imwrite(output_path, frame)

            print(f"\n[{i}/{len(image_paths)}] {os.path.basename(img_path)}")
            if events:
                for event in events:
                    label = EVENT_LABELS.get(event.event_type, event.event_type.value)
                    print(f"  [{event.severity.upper()}] {label}: {event.description}")
            else:
                print("  未检测到交通事件")

            # 显示
            cv2.imshow('Traffic Event Detection', frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

        self._cleanup()

    def _cleanup(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 50)
        print("检测结束 - 事件统计:")
        print("-" * 30)
        for event_type, count in self.total_events.items():
            try:
                label = EVENT_LABELS[EventType(event_type)]
            except (ValueError, KeyError):
                label = event_type
            print(f"  {label}: {count}次")
        print(f"  总计: {sum(self.total_events.values())}次")
        print(f"  处理帧数: {self.frame_count}")

    def run(self):
        """启动检测"""
        if not self.load_model():
            return
        if not self.init_source():
            return

        if self.source == 'images':
            self.run_images()
        else:
            self.run_camera_or_video()


from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description='交通事件检测系统')
    parser.add_argument('--source', type=str, default='camera',
                        choices=['camera', 'video', 'images'],
                        help='输入源: camera(摄像头), video(视频), images(图片目录)')
    parser.add_argument('--input', type=str, default=None,
                        help='视频文件路径或图片目录路径')
    parser.add_argument('--model', type=str, default='models/yolov8n.pt',
                        help='YOLO模型路径')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='检测置信度阈值 (0.1-0.9)')
    parser.add_argument('--direction', type=str, default='right',
                        choices=['right', 'left', 'up', 'down'],
                        help='车辆正常行驶方向')

    args = parser.parse_args()

    system = TrafficDetectionSystem(
        model_path=args.model,
        source=args.source,
        input_path=args.input,
        confidence=args.conf,
        direction=args.direction,
    )
    system.run()


if __name__ == "__main__":
    main()
