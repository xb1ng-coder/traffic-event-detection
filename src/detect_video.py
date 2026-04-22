"""
视频交通事件检测脚本
专门用于检测 data/videos/ 目录下的视频文件，输出检测结果和统计报告

使用方法:
  python src/detect_video.py
  python src/detect_video.py --input data/videos/cctv052x2004080517x01659.mp4
  python src/detect_video.py --direction down    # 俯视角摄像头，车辆向下行驶
  python src/detect_video.py --save-output       # 保存标注后的输出视频
  python src/detect_video.py --no-display        # 不显示窗口，仅输出统计
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import glob
from collections import defaultdict

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
    'danger': (0, 0, 255),
    'warning': (0, 165, 255),
    'info': (0, 255, 255),
}


class VideoTrafficDetector:
    """视频交通事件检测器"""

    def __init__(self, model_path, confidence=0.3, direction='right',
                 save_output=False, display=True, skip_frames=0):
        self.model_path = model_path
        self.confidence = confidence
        self.direction = direction
        self.save_output = save_output
        self.display = display
        self.skip_frames = skip_frames

        # 项目根目录（src 的上一级）
        self.project_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

        self.model = None
        self.event_detector = None

    def load_model(self):
        """加载YOLO模型"""
        print("\n[1/3] 加载模型...")
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"  本地模型: {self.model_path}")
            else:
                print("  下载YOLOv8n模型...")
                self.model = YOLO('yolov8n.pt')
                # 保存到项目根目录的 models/ 下
                save_dir = os.path.dirname(self.model_path)
                os.makedirs(save_dir, exist_ok=True)
                self.model.save(self.model_path)
                print(f"  已保存: {self.model_path}")
        except Exception as e:
            print(f"  加载失败: {e}")
            return False

        # 预热
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)
        print("  模型就绪")
        return True

    def _parse_yolo_results(self, results):
        """解析YOLO检测结果"""
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 轨迹线
            if len(track.positions) >= 2:
                pts = np.array(track.positions[-30:], dtype=np.int32)
                cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)

            # 标签
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

    def _draw_events_panel(self, frame, events, frame_idx):
        """绘制事件面板"""
        if not events:
            return frame

        h, w = frame.shape[:2]
        panel_w = 380
        panel_h = 45 + len(events) * 30
        panel_x = w - panel_w - 10
        panel_y = 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, f"EVENTS (Frame {frame_idx})", (panel_x + 10, panel_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        y_off = 48
        for event in events:
            color = SEVERITY_COLORS.get(event.severity, (255, 255, 255))
            label = EVENT_LABELS.get(event.event_type, event.event_type.value)
            text = f"[{event.severity.upper()}] {label}"

            cv2.rectangle(frame, (panel_x + 5, panel_y + y_off - 12),
                          (panel_x + 9, panel_y + y_off + 8), color, -1)
            cv2.putText(frame, text, (panel_x + 14, panel_y + y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 事件位置标记
            loc_x = int(event.location[0] * w)
            loc_y = int(event.location[1] * h)
            cv2.drawMarker(frame, (loc_x, loc_y), color, cv2.MARKER_CROSS, 30, 2)

            y_off += 30

        return frame

    def _draw_stats_panel(self, frame, fps, frame_idx, total_frames, event_counts):
        """绘制统计面板"""
        panel_w = 280
        panel_h = 200
        panel_x, panel_y = 10, 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        y = panel_y + 25
        cv2.putText(frame, "Traffic Event Detection", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 25
        cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        y += 22
        progress = frame_idx / total_frames * 100 if total_frames > 0 else 0
        cv2.putText(frame, f"Progress: {frame_idx}/{total_frames} ({progress:.1f}%)",
                    (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 22
        cv2.putText(frame, f"Direction: {self.direction}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25

        tracked = len(self.event_detector.tracker.tracks)
        cv2.putText(frame, f"Tracked: {tracked}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 1)
        y += 22

        total_events = sum(event_counts.values())
        cv2.putText(frame, f"Events: {total_events}", (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        y += 22

        # 进度条
        bar_x = panel_x + 10
        bar_w = panel_w - 20
        bar_h = 8
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_h), (100, 100, 100), -1)
        fill_w = int(bar_w * progress / 100)
        cv2.rectangle(frame, (bar_x, y), (bar_x + fill_w, y + bar_h), (0, 255, 0), -1)

        return frame

    def _draw_road_region(self, frame):
        """绘制道路区域"""
        r = self.event_detector.road_region
        x1 = int(r[0] * self.event_detector.frame_width)
        y1 = int(r[1] * self.event_detector.frame_height)
        x2 = int(r[2] * self.event_detector.frame_width)
        y2 = int(r[3] * self.event_detector.frame_height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
        cv2.putText(frame, "Road", (x1 + 5, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        return frame

    def process_video(self, video_path):
        """
        处理单个视频文件

        Returns:
            dict: 检测结果统计
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  无法打开: {video_path}")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n{'='*60}")
        print(f"视频: {os.path.basename(video_path)}")
        print(f"分辨率: {width}x{height} | 帧率: {fps:.0f} | 总帧数: {total_frames}")
        print(f"时长: {total_frames/fps:.1f}秒")
        print(f"行驶方向: {self.direction} | 置信度: {self.confidence}")
        if self.skip_frames > 0:
            print(f"跳帧: 每隔{self.skip_frames}帧处理1帧")
        print(f"{'='*60}")

        # 初始化事件检测器
        self.event_detector = TrafficEventDetector(
            frame_width=width, frame_height=height, fps=fps
        )
        self.event_detector.set_normal_direction(self.direction)

        # 输出视频
        writer = None
        if self.save_output:
            os.makedirs(os.path.join(self.project_root, "results"), exist_ok=True)
            basename = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(self.project_root, f"results/{basename}_detected.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                print(f"  警告: 无法创建输出视频，尝试XVID编码...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                output_path = os.path.join(self.project_root, f"results/{basename}_detected.avi")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"  输出视频: {output_path}")

        # 统计
        event_counts = defaultdict(int)
        frame_event_log = []  # 每帧事件记录
        frame_idx = 0
        processed_frames = 0
        start_time = time.time()
        current_fps = 0
        paused = False

        print("\n开始检测... (q=退出, p=暂停, s=截图)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # 跳帧处理
            if self.skip_frames > 0 and (frame_idx - 1) % (self.skip_frames + 1) != 0:
                continue

            processed_frames += 1

            # YOLO检测
            results = self.model(frame, conf=self.confidence, verbose=False)
            detections = self._parse_yolo_results(results)

            # 事件检测
            events = self.event_detector.detect(detections)

            # 记录事件
            if events:
                frame_events_info = {
                    'frame': frame_idx,
                    'time': frame_idx / fps,
                    'events': []
                }
                for event in events:
                    event_counts[event.event_type.value] += 1
                    label = EVENT_LABELS.get(event.event_type, event.event_type.value)
                    frame_events_info['events'].append({
                        'type': event.event_type.value,
                        'label': label,
                        'severity': event.severity,
                        'description': event.description,
                    })
                    # 控制台输出
                    time_sec = frame_idx / fps
                    print(f"  [{time_sec:6.1f}s] 帧{frame_idx:>5d} | "
                          f"[{event.severity.upper()}] {label}: {event.description}")

                frame_event_log.append(frame_events_info)

            # 计算FPS
            elapsed = time.time() - start_time
            current_fps = processed_frames / elapsed if elapsed > 0 else 0

            # 绘制可视化
            tracks = self.event_detector.tracker.tracks
            frame = self._draw_tracks(frame, tracks)
            frame = self._draw_road_region(frame)
            frame = self._draw_events_panel(frame, events, frame_idx)
            frame = self._draw_stats_panel(frame, current_fps, frame_idx, total_frames, event_counts)

            # 保存输出视频
            if writer:
                writer.write(frame)

            # 显示
            if self.display:
                cv2.imshow('Traffic Event Detection', frame)

                if not paused:
                    wait_key = 1
                else:
                    wait_key = 0

                key = cv2.waitKey(wait_key) & 0xFF
                if key == ord('q'):
                    print("\n  用户中断")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"  {'暂停' if paused else '继续'}")
                elif key == ord('s'):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(self.project_root, f"results/capture_{ts}.jpg")
                    cv2.imwrite(path, frame)
                    print(f"  截图: {path}")

            # 进度提示
            if processed_frames % 50 == 0:
                progress = frame_idx / total_frames * 100
                print(f"  进度: {frame_idx}/{total_frames} ({progress:.1f}%) FPS:{current_fps:.1f}")

        # 清理
        cap.release()
        if writer:
            writer.release()
        if self.display:
            cv2.destroyAllWindows()

        # 统计结果
        total_time = time.time() - start_time
        result = {
            'video': os.path.basename(video_path),
            'resolution': f"{width}x{height}",
            'fps': fps,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'duration': total_frames / fps,
            'processing_time': total_time,
            'avg_fps': processed_frames / total_time if total_time > 0 else 0,
            'event_counts': dict(event_counts),
            'total_events': sum(event_counts.values()),
            'frame_event_log': frame_event_log,
        }

        return result

    def print_report(self, result):
        """打印检测报告"""
        if not result:
            return

        print(f"\n{'='*60}")
        print(f"检测报告: {result['video']}")
        print(f"{'='*60}")
        print(f"视频信息:")
        print(f"  分辨率: {result['resolution']}")
        print(f"  帧率: {result['fps']:.0f} FPS")
        print(f"  时长: {result['duration']:.1f} 秒 ({result['total_frames']} 帧)")
        print(f"  处理帧数: {result['processed_frames']}")
        print(f"\n处理性能:")
        print(f"  耗时: {result['processing_time']:.1f} 秒")
        print(f"  平均FPS: {result['avg_fps']:.1f}")
        print(f"\n事件统计:")
        print(f"  事件总数: {result['total_events']}")

        if result['event_counts']:
            for event_type, count in sorted(result['event_counts'].items(),
                                            key=lambda x: x[1], reverse=True):
                try:
                    label = EVENT_LABELS[EventType(event_type)]
                except (ValueError, KeyError):
                    label = event_type
                bar = '#' * min(count, 50)
                print(f"  {label:12s}: {count:>4d}次  {bar}")
        else:
            print("  未检测到交通事件")

        # 事件时间线
        if result['frame_event_log']:
            print(f"\n事件时间线:")
            for fe in result['frame_event_log'][:20]:  # 最多显示20条
                events_str = ", ".join(e['label'] for e in fe['events'])
                print(f"  {fe['time']:6.1f}s (帧{fe['frame']}) -> {events_str}")
            if len(result['frame_event_log']) > 20:
                print(f"  ... 共{len(result['frame_event_log'])}条记录")

        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='视频交通事件检测')
    parser.add_argument('--input', type=str, default=None,
                        help='视频文件路径 (默认检测data/videos/下所有视频)')
    # 默认模型路径：项目根目录下的 models/yolov8n.pt
    default_model = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'yolov8n.pt'))
    parser.add_argument('--model', type=str, default=default_model,
                        help='YOLO模型路径')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='检测置信度阈值 (0.1-0.9)')
    parser.add_argument('--direction', type=str, default='right',
                        choices=['right', 'left', 'up', 'down'],
                        help='车辆正常行驶方向')
    parser.add_argument('--save-output', action='store_true',
                        help='保存标注后的输出视频')
    parser.add_argument('--no-display', action='store_true',
                        help='不显示检测窗口(加速处理)')
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='跳帧数(0=不跳帧, 1=每隔1帧处理, 2=每隔2帧处理)')

    args = parser.parse_args()

    # 项目根目录
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

    # 确定视频列表
    if args.input:
        video_paths = [args.input]
    else:
        video_dir = os.path.join(project_root, 'data', 'videos')
        video_paths = []
        for fmt in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']:
            video_paths.extend(glob.glob(os.path.join(video_dir, fmt)))
        video_paths = sorted(video_paths)

        if not video_paths:
            print(f"未找到视频文件: {video_dir}")
            print("请将视频放入 data/videos/ 目录，或使用 --input 指定路径")
            return

    print(f"交通事件检测系统")
    print(f"找到 {len(video_paths)} 个视频文件")

    # 初始化检测器
    detector = VideoTrafficDetector(
        model_path=args.model,
        confidence=args.conf,
        direction=args.direction,
        save_output=args.save_output,
        display=not args.no_display,
        skip_frames=args.skip_frames,
    )

    if not detector.load_model():
        return

    # 逐个处理视频
    all_results = []
    for i, video_path in enumerate(video_paths, 1):
        print(f"\n[{i}/{len(video_paths)}] 处理: {os.path.basename(video_path)}")
        result = detector.process_video(video_path)
        if result:
            detector.print_report(result)
            all_results.append(result)

    # 汇总报告
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("汇总报告")
        print(f"{'='*60}")
        for r in all_results:
            total = r['total_events']
            print(f"  {r['video']}: {total}个事件, "
                  f"处理速度{r['avg_fps']:.1f}FPS")
        total_all = sum(r['total_events'] for r in all_results)
        print(f"\n  全部事件总计: {total_all}个")

    print("\n检测完成!")


if __name__ == "__main__":
    main()
