"""
交通事件检测模块
基于YOLO目标检测结果，通过时空分析实现交通事件检测

支持检测的事件类型:
1. 车辆停止/违停 (Vehicle Stopped)
2. 车辆逆行 (Wrong-way Driving)
3. 行人闯入车道 (Pedestrian Intrusion)
4. 交通拥堵 (Traffic Congestion)
5. 车辆碰撞/事故 (Vehicle Collision)
6. 车辆超速 (Speeding)
7. 车辆缓慢行驶 (Slow Moving)
8. 道路遗洒物 (Debris on Road)
"""

import numpy as np
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class EventType(Enum):
    """交通事件类型"""
    VEHICLE_STOPPED = "vehicle_stopped"           # 车辆停止/违停
    WRONG_WAY = "wrong_way"                       # 逆行
    PEDESTRIAN_INTRUSION = "pedestrian_intrusion"  # 行人闯入车道
    CONGESTION = "congestion"                      # 交通拥堵
    COLLISION = "collision"                        # 碰撞/事故
    SPEEDING = "speeding"                          # 超速
    SLOW_MOVING = "slow_moving"                    # 缓慢行驶
    DEBRIS = "debris"                              # 遗洒物


# 事件严重等级
SEVERITY_LEVELS = {
    EventType.VEHICLE_STOPPED: "warning",
    EventType.WRONG_WAY: "danger",
    EventType.PEDESTRIAN_INTRUSION: "danger",
    EventType.CONGESTION: "info",
    EventType.COLLISION: "danger",
    EventType.SPEEDING: "warning",
    EventType.SLOW_MOVING: "info",
    EventType.DEBRIS: "warning",
}

# 事件中文描述
EVENT_LABELS = {
    EventType.VEHICLE_STOPPED: "车辆停止/违停",
    EventType.WRONG_WAY: "车辆逆行",
    EventType.PEDESTRIAN_INTRUSION: "行人闯入车道",
    EventType.CONGESTION: "交通拥堵",
    EventType.COLLISION: "疑似碰撞/事故",
    EventType.SPEEDING: "车辆超速",
    EventType.SLOW_MOVING: "车辆缓行",
    EventType.DEBRIS: "道路遗洒物",
}


@dataclass
class TrackedObject:
    """被追踪的目标"""
    track_id: int
    class_name: str
    bbox: List[float]       # [x1, y1, x2, y2]
    confidence: float
    center: Tuple[float, float]  # 中心点 (cx, cy)
    frame_count: int = 0          # 追踪帧数
    last_seen: float = 0.0        # 上次出现时间
    positions: List[Tuple[float, float]] = field(default_factory=list)  # 历史中心点
    speeds: List[float] = field(default_factory=list)                  # 历史速度
    stopped_frames: int = 0       # 停止帧数计数


@dataclass
class TrafficEvent:
    """交通事件"""
    event_type: EventType
    severity: str
    description: str
    location: Tuple[float, float]     # 事件发生位置 (归一化坐标)
    track_ids: List[int]              # 相关目标ID
    confidence: float = 1.0           # 事件置信度
    timestamp: float = 0.0
    frame_idx: int = 0


class SimpleTracker:
    """
    简单的目标追踪器（基于IoU匹配）
    无需依赖DeepSORT等外部库，适合轻量化部署
    """

    def __init__(self, max_disappeared=30, iou_threshold=0.3):
        self.next_id = 0
        self.tracks: Dict[int, TrackedObject] = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold

    def _compute_iou(self, box1, box2):
        """计算两个框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def update(self, detections: List[Tuple[str, List[float], float]]):
        """
        更新追踪状态

        Args:
            detections: [(class_name, [x1,y1,x2,y2], confidence), ...]
        
        Returns:
            dict: {track_id: TrackedObject}
        """
        current_time = time.time()

        # 如果没有现有追踪，全部注册新目标
        if not self.tracks:
            for cls_name, bbox, conf in detections:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                track = TrackedObject(
                    track_id=self.next_id,
                    class_name=cls_name,
                    bbox=bbox,
                    confidence=conf,
                    center=(cx, cy),
                    frame_count=1,
                    last_seen=current_time,
                    positions=[(cx, cy)],
                    speeds=[],
                    stopped_frames=0,
                )
                self.tracks[self.next_id] = track
                self.next_id += 1
            return self.tracks

        # 匹配现有追踪和新检测
        track_ids = list(self.tracks.keys())
        used_detections = set()
        used_tracks = set()

        # 构建IoU代价矩阵
        iou_matrix = np.zeros((len(track_ids), len(detections)))
        for i, tid in enumerate(track_ids):
            for j, (cls_name, bbox, conf) in enumerate(detections):
                # 只匹配相同类别
                if self.tracks[tid].class_name == cls_name:
                    iou_matrix[i, j] = self._compute_iou(self.tracks[tid].bbox, bbox)

        # 贪心匹配
        while True:
            max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_iou_idx]
            if max_iou < self.iou_threshold:
                break

            i, j = max_iou_idx
            tid = track_ids[i]
            cls_name, bbox, conf = detections[j]

            # 更新已匹配的追踪
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            track = self.tracks[tid]

            # 计算速度（像素/帧）
            if track.positions:
                prev_cx, prev_cy = track.positions[-1]
                speed = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                track.speeds.append(speed)
                # 保留最近30帧速度
                if len(track.speeds) > 30:
                    track.speeds = track.speeds[-30:]

            track.bbox = bbox
            track.confidence = conf
            track.center = (cx, cy)
            track.frame_count += 1
            track.last_seen = current_time
            track.positions.append((cx, cy))
            # 保留最近60帧位置
            if len(track.positions) > 60:
                track.positions = track.positions[-60:]

            # 判断是否停止
            avg_speed = np.mean(track.speeds[-5:]) if track.speeds else 0
            if avg_speed < 2.0:  # 阈值：几乎不动
                track.stopped_frames += 1
            else:
                track.stopped_frames = 0

            used_detections.add(j)
            used_tracks.add(tid)
            iou_matrix[i, j] = 0  # 清除已匹配

        # 未匹配的检测 -> 注册新目标
        for j, (cls_name, bbox, conf) in enumerate(detections):
            if j not in used_detections:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                track = TrackedObject(
                    track_id=self.next_id,
                    class_name=cls_name,
                    bbox=bbox,
                    confidence=conf,
                    center=(cx, cy),
                    frame_count=1,
                    last_seen=current_time,
                    positions=[(cx, cy)],
                    speeds=[],
                    stopped_frames=0,
                )
                self.tracks[self.next_id] = track
                self.next_id += 1

        # 移除长时间消失的追踪
        to_remove = []
        for tid, track in self.tracks.items():
            if current_time - track.last_seen > 2.0:  # 2秒未出现
                to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

        return self.tracks


class TrafficEventDetector:
    """
    交通事件检测器
    基于YOLO检测结果 + 目标追踪 + 规则引擎
    """

    # 交通相关类别
    VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
    PERSON_CLASSES = {'person'}
    ALL_TRAFFIC_CLASSES = VEHICLE_CLASSES | PERSON_CLASSES

    def __init__(self, frame_width=1280, frame_height=720, fps=30):
        """
        Args:
            frame_width: 视频帧宽度
            frame_height: 视频帧高度
            fps: 视频帧率
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.frame_idx = 0

        # 追踪器
        self.tracker = SimpleTracker(max_disappeared=int(fps * 2))

        # 事件配置参数
        self.config = {
            # 停车检测
            'stop_min_frames': int(fps * 3),     # 连续停止3秒判定为停车
            'stop_speed_threshold': 2.0,          # 速度阈值（像素/帧）

            # 逆行检测
            'wrong_way_min_frames': int(fps * 1.5),  # 1.5秒
            'wrong_way_movement_threshold': 30,    # 逆行最小位移（像素）

            # 拥堵检测
            'congestion_vehicle_count': 5,         # 拥堵车辆数阈值
            'congestion_area_ratio': 0.4,          # 车辆在画面中的面积占比

            # 碰撞检测
            'collision_iou_threshold': 0.1,         # 重叠IoU阈值
            'collision_speed_threshold': 5.0,       # 碰撞前速度阈值

            # 超速检测
            'speeding_threshold': 25.0,             # 超速阈值（像素/帧）

            # 缓行检测
            'slow_speed_threshold': 5.0,            # 缓行速度阈值
            'slow_min_frames': int(fps * 2),        # 持续2秒

            # 行人闯入
            'pedestrian_invasion_frames': int(fps * 1),  # 1秒

            # 事件冷却时间（秒）- 避免同一事件反复报警
            'event_cooldown': 5.0,
        }

        # 事件历史与冷却
        self.active_events: List[TrafficEvent] = []
        self.event_history: List[TrafficEvent] = []
        self.last_event_time: Dict[EventType, float] = {}

        # 区域定义（归一化坐标）- 可根据实际场景调整
        # 默认车道区域：画面下半部分中间区域
        self.road_region = (0.1, 0.4, 0.9, 0.95)  # (x1_norm, y1_norm, x2_norm, y2_norm)

        # 方向定义：正常行驶方向（像素位移方向）
        # 默认假设车辆从左向右行驶（适用于侧视角摄像头）
        # 如果是俯视角/前视角，需要调整
        self.normal_direction = 'right'  # 'right', 'left', 'up', 'down'

    def set_road_region(self, x1, y1, x2, y2):
        """设置道路区域（归一化坐标 0-1）"""
        self.road_region = (x1, y1, x2, y2)

    def set_normal_direction(self, direction: str):
        """设置正常行驶方向"""
        assert direction in ('right', 'left', 'up', 'down')
        self.normal_direction = direction

    def _is_in_road(self, cx, cy):
        """判断点是否在道路区域内"""
        nx = cx / self.frame_width
        ny = cy / self.frame_height
        return (self.road_region[0] <= nx <= self.road_region[2] and
                self.road_region[1] <= ny <= self.road_region[3])

    def _get_displacement_direction(self, positions):
        """
        根据历史位置判断移动方向

        Returns:
            (dx, dy): 总位移向量
        """
        if len(positions) < 2:
            return (0, 0)
        # 使用最近几帧的位移
        recent = positions[-min(10, len(positions)):]
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        return (dx, dy)

    def _is_wrong_direction(self, dx, dy):
        """判断位移是否为逆行方向"""
        threshold = self.config['wrong_way_movement_threshold']
        if self.normal_direction == 'right' and dx < -threshold:
            return True
        elif self.normal_direction == 'left' and dx > threshold:
            return True
        elif self.normal_direction == 'down' and dy < -threshold:
            return True
        elif self.normal_direction == 'up' and dy > threshold:
            return True
        return False

    def _compute_overlap_ratio(self, box1, box2):
        """计算两个框的重叠面积比（相对于较小框）"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        min_area = min(area1, area2)

        return inter_area / min_area if min_area > 0 else 0

    def _is_event_cooled(self, event_type: EventType) -> bool:
        """检查事件是否在冷却期内"""
        if event_type not in self.last_event_time:
            return True
        elapsed = time.time() - self.last_event_time[event_type]
        return elapsed > self.config['event_cooldown']

    def _add_event(self, event: TrafficEvent):
        """添加事件（带冷却控制）"""
        if self._is_event_cooled(event.event_type):
            event.timestamp = time.time()
            event.frame_idx = self.frame_idx
            self.active_events.append(event)
            self.event_history.append(event)
            self.last_event_time[event.event_type] = event.timestamp

    def _detect_stopped_vehicles(self, tracks: Dict[int, TrackedObject]):
        """检测停止/违停的车辆"""
        for tid, track in tracks.items():
            if track.class_name not in self.VEHICLE_CLASSES:
                continue

            # 需要足够的追踪帧数
            if track.frame_count < self.config['stop_min_frames']:
                continue

            # 连续停止帧数超过阈值
            if track.stopped_frames >= self.config['stop_min_frames']:
                # 确认在道路区域内
                if self._is_in_road(track.center[0], track.center[1]):
                    avg_speed = np.mean(track.speeds[-10:]) if track.speeds else 0
                    self._add_event(TrafficEvent(
                        event_type=EventType.VEHICLE_STOPPED,
                        severity=SEVERITY_LEVELS[EventType.VEHICLE_STOPPED],
                        description=f"{track.class_name} ID:{tid} 停留超过{track.stopped_frames // self.fps:.0f}秒",
                        location=(track.center[0] / self.frame_width,
                                  track.center[1] / self.frame_height),
                        track_ids=[tid],
                        confidence=min(1.0, track.stopped_frames / (self.config['stop_min_frames'] * 2)),
                    ))

    def _detect_wrong_way(self, tracks: Dict[int, TrackedObject]):
        """检测逆行车辆"""
        for tid, track in tracks.items():
            if track.class_name not in self.VEHICLE_CLASSES:
                continue

            if len(track.positions) < self.config['wrong_way_min_frames']:
                continue

            dx, dy = self._get_displacement_direction(track.positions)

            if self._is_wrong_direction(dx, dy):
                # 计算位移幅度
                displacement = np.sqrt(dx ** 2 + dy ** 2)
                if displacement > self.config['wrong_way_movement_threshold']:
                    self._add_event(TrafficEvent(
                        event_type=EventType.WRONG_WAY,
                        severity=SEVERITY_LEVELS[EventType.WRONG_WAY],
                        description=f"{track.class_name} ID:{tid} 疑似逆行",
                        location=(track.center[0] / self.frame_width,
                                  track.center[1] / self.frame_height),
                        track_ids=[tid],
                        confidence=min(1.0, displacement / 100),
                    ))

    def _detect_pedestrian_intrusion(self, tracks: Dict[int, TrackedObject]):
        """检测行人闯入车道"""
        for tid, track in tracks.items():
            if track.class_name not in self.PERSON_CLASSES:
                continue

            if track.frame_count < self.config['pedestrian_invasion_frames']:
                continue

            # 行人在道路区域内
            if self._is_in_road(track.center[0], track.center[1]):
                # 检查附近是否有车辆
                nearby_vehicles = []
                for vid, vtrack in tracks.items():
                    if vtrack.class_name in self.VEHICLE_CLASSES:
                        dist = np.sqrt(
                            (track.center[0] - vtrack.center[0]) ** 2 +
                            (track.center[1] - vtrack.center[1]) ** 2
                        )
                        if dist < 200:  # 200像素范围内
                            nearby_vehicles.append(vid)

                self._add_event(TrafficEvent(
                    event_type=EventType.PEDESTRIAN_INTRUSION,
                    severity=SEVERITY_LEVELS[EventType.PEDESTRIAN_INTRUSION],
                    description=f"行人 ID:{tid} 在车道内{'，附近有车辆' if nearby_vehicles else ''}",
                    location=(track.center[0] / self.frame_width,
                              track.center[1] / self.frame_height),
                    track_ids=[tid] + nearby_vehicles,
                    confidence=0.9 if nearby_vehicles else 0.6,
                ))

    def _detect_congestion(self, tracks: Dict[int, TrackedObject]):
        """检测交通拥堵"""
        vehicles_in_road = []
        for tid, track in tracks.items():
            if track.class_name in self.VEHICLE_CLASSES and self._is_in_road(track.center[0], track.center[1]):
                vehicles_in_road.append(track)

        vehicle_count = len(vehicles_in_road)
        if vehicle_count >= self.config['congestion_vehicle_count']:
            # 计算车辆面积占比
            total_vehicle_area = 0
            road_area = ((self.road_region[2] - self.road_region[0]) * self.frame_width *
                         (self.road_region[3] - self.road_region[1]) * self.frame_height)

            for v in vehicles_in_road:
                w = v.bbox[2] - v.bbox[0]
                h = v.bbox[3] - v.bbox[1]
                total_vehicle_area += w * h

            area_ratio = total_vehicle_area / road_area if road_area > 0 else 0

            # 计算平均速度
            avg_speeds = []
            for v in vehicles_in_road:
                if v.speeds:
                    avg_speeds.append(np.mean(v.speeds[-10:]))
            mean_speed = np.mean(avg_speeds) if avg_speeds else 0

            # 车辆数量多 + 速度低 = 拥堵
            if vehicle_count >= self.config['congestion_vehicle_count'] and mean_speed < self.config['slow_speed_threshold']:
                self._add_event(TrafficEvent(
                    event_type=EventType.CONGESTION,
                    severity=SEVERITY_LEVELS[EventType.CONGESTION],
                    description=f"检测到{vehicle_count}辆车在车道内，平均速度{mean_speed:.1f}，疑似拥堵",
                    location=(0.5, 0.5),  # 画面中心
                    track_ids=[v.track_id for v in vehicles_in_road],
                    confidence=min(1.0, vehicle_count / 10),
                ))

    def _detect_collision(self, tracks: Dict[int, TrackedObject]):
        """检测碰撞/事故"""
        vehicle_tracks = {tid: t for tid, t in tracks.items()
                         if t.class_name in self.VEHICLE_CLASSES and len(t.speeds) >= 3}
        tids = list(vehicle_tracks.keys())

        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                t1 = vehicle_tracks[tids[i]]
                t2 = vehicle_tracks[tids[j]]

                # 检查两车是否有显著重叠
                overlap = self._compute_overlap_ratio(t1.bbox, t2.bbox)
                if overlap > self.config['collision_iou_threshold']:
                    # 检查碰撞前是否高速
                    s1 = np.mean(t1.speeds[-5:]) if t1.speeds else 0
                    s2 = np.mean(t2.speeds[-5:]) if t2.speeds else 0

                    # 检查碰撞后是否减速/停止
                    s1_recent = np.mean(t1.speeds[-3:]) if len(t1.speeds) >= 3 else 0
                    s2_recent = np.mean(t2.speeds[-3:]) if len(t2.speeds) >= 3 else 0

                    # 碰撞模式：之前高速 + 重叠 + 之后减速
                    if (s1 > self.config['collision_speed_threshold'] or
                            s2 > self.config['collision_speed_threshold']):
                        if s1_recent < s1 * 0.3 or s2_recent < s2 * 0.3 or \
                           t1.stopped_frames > 3 or t2.stopped_frames > 3:
                            # 合并位置
                            cx = (t1.center[0] + t2.center[0]) / 2
                            cy = (t1.center[1] + t2.center[1]) / 2
                            self._add_event(TrafficEvent(
                                event_type=EventType.COLLISION,
                                severity=SEVERITY_LEVELS[EventType.COLLISION],
                                description=f"{t1.class_name}与{t2.class_name}疑似碰撞",
                                location=(cx / self.frame_width, cy / self.frame_height),
                                track_ids=[t1.track_id, t2.track_id],
                                confidence=min(1.0, overlap * 2),
                            ))

    def _detect_speeding(self, tracks: Dict[int, TrackedObject]):
        """检测超速车辆"""
        for tid, track in tracks.items():
            if track.class_name not in self.VEHICLE_CLASSES:
                continue
            if len(track.speeds) < 5:
                continue

            avg_speed = np.mean(track.speeds[-10:])
            if avg_speed > self.config['speeding_threshold']:
                self._add_event(TrafficEvent(
                    event_type=EventType.SPEEDING,
                    severity=SEVERITY_LEVELS[EventType.SPEEDING],
                    description=f"{track.class_name} ID:{tid} 疑似超速(速度:{avg_speed:.1f})",
                    location=(track.center[0] / self.frame_width,
                              track.center[1] / self.frame_height),
                    track_ids=[tid],
                    confidence=min(1.0, avg_speed / (self.config['speeding_threshold'] * 2)),
                ))

    def _detect_slow_moving(self, tracks: Dict[int, TrackedObject]):
        """检测缓行车辆"""
        for tid, track in tracks.items():
            if track.class_name not in self.VEHICLE_CLASSES:
                continue
            if track.frame_count < self.config['slow_min_frames']:
                continue

            if len(track.speeds) < 5:
                continue

            avg_speed = np.mean(track.speeds[-10:])
            if (self.config['stop_speed_threshold'] < avg_speed < self.config['slow_speed_threshold']
                    and track.stopped_frames < self.config['stop_min_frames']):
                if self._is_in_road(track.center[0], track.center[1]):
                    self._add_event(TrafficEvent(
                        event_type=EventType.SLOW_MOVING,
                        severity=SEVERITY_LEVELS[EventType.SLOW_MOVING],
                        description=f"{track.class_name} ID:{tid} 缓慢行驶(速度:{avg_speed:.1f})",
                        location=(track.center[0] / self.frame_width,
                                  track.center[1] / self.frame_height),
                        track_ids=[tid],
                        confidence=0.7,
                    ))

    def detect(self, detections: List[Tuple[str, List[float], float]]) -> List[TrafficEvent]:
        """
        执行交通事件检测

        Args:
            detections: YOLO检测结果列表 [(class_name, [x1,y1,x2,y2], confidence), ...]

        Returns:
            当前帧检测到的事件列表
        """
        self.frame_idx += 1
        self.active_events = []

        # 过滤交通相关目标
        traffic_dets = [d for d in detections if d[0] in self.ALL_TRAFFIC_CLASSES]

        if not traffic_dets:
            return self.active_events

        # 更新追踪器
        tracks = self.tracker.update(traffic_dets)

        # 执行各项事件检测
        self._detect_stopped_vehicles(tracks)
        self._detect_wrong_way(tracks)
        self._detect_pedestrian_intrusion(tracks)
        self._detect_congestion(tracks)
        self._detect_collision(tracks)
        self._detect_speeding(tracks)
        self._detect_slow_moving(tracks)

        return self.active_events

    def get_event_summary(self) -> Dict:
        """获取事件统计摘要"""
        summary = defaultdict(int)
        for event in self.event_history:
            summary[event.event_type.value] += 1
        return dict(summary)

    def get_tracked_objects_info(self) -> List[Dict]:
        """获取当前追踪目标信息"""
        info = []
        for tid, track in self.tracker.tracks.items():
            avg_speed = np.mean(track.speeds[-10:]) if track.speeds else 0
            info.append({
                'track_id': track.track_id,
                'class_name': track.class_name,
                'confidence': track.confidence,
                'center': track.center,
                'avg_speed': round(avg_speed, 2),
                'frame_count': track.frame_count,
                'stopped_frames': track.stopped_frames,
                'is_stopped': track.stopped_frames >= self.config['stop_min_frames'],
            })
        return info
