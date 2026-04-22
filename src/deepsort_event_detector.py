"""
基于DeepSORT的交通事件检测模块

相比 SimpleTracker (IoU匹配)，DeepSORT 引入外观特征(ReID)，
在目标被遮挡后重新出现时能更好地保持ID一致性，追踪更稳定。

依赖: pip install deep-sort-realtime

支持检测的事件类型:
1. 车辆停止/违停 (Vehicle Stopped)
2. 车辆逆行 (Wrong-way Driving)
3. 行人闯入车道 (Pedestrian Intrusion)
4. 交通拥堵 (Traffic Congestion)
5. 车辆碰撞/事故 (Vehicle Collision)
6. 车辆超速 (Speeding)
7. 车辆缓慢行驶 (Slow Moving)
8. 车辆紧急变道 (Sudden Lane Change)
"""

import numpy as np
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

from deep_sort_realtime.deepsort_tracker import DeepSort


class EventType(Enum):
    """交通事件类型"""
    VEHICLE_STOPPED = "vehicle_stopped"
    WRONG_WAY = "wrong_way"
    PEDESTRIAN_INTRUSION = "pedestrian_intrusion"
    CONGESTION = "congestion"
    COLLISION = "collision"
    SPEEDING = "speeding"
    SLOW_MOVING = "slow_moving"
    SUDDEN_LANE_CHANGE = "sudden_lane_change"


SEVERITY_LEVELS = {
    EventType.VEHICLE_STOPPED: "warning",
    EventType.WRONG_WAY: "danger",
    EventType.PEDESTRIAN_INTRUSION: "danger",
    EventType.CONGESTION: "info",
    EventType.COLLISION: "danger",
    EventType.SPEEDING: "warning",
    EventType.SLOW_MOVING: "info",
    EventType.SUDDEN_LANE_CHANGE: "warning",
}

EVENT_LABELS = {
    EventType.VEHICLE_STOPPED: "车辆停止/违停",
    EventType.WRONG_WAY: "车辆逆行",
    EventType.PEDESTRIAN_INTRUSION: "行人闯入车道",
    EventType.CONGESTION: "交通拥堵",
    EventType.COLLISION: "疑似碰撞/事故",
    EventType.SPEEDING: "车辆超速",
    EventType.SLOW_MOVING: "车辆缓行",
    EventType.SUDDEN_LANE_CHANGE: "紧急变道",
}


@dataclass
class DeepTrack:
    """DeepSORT追踪目标的扩展信息"""
    track_id: int
    class_name: str
    bbox: List[float]            # [x1, y1, x2, y2]
    confidence: float
    center: Tuple[float, float]
    frame_count: int = 0
    last_seen: float = 0.0
    positions: List[Tuple[float, float]] = field(default_factory=list)
    speeds: List[float] = field(default_factory=list)
    stopped_frames: int = 0
    lateral_positions: List[float] = field(default_factory=list)  # 横向位置历史(用于变道检测)


@dataclass
class TrafficEvent:
    """交通事件"""
    event_type: EventType
    severity: str
    description: str
    location: Tuple[float, float]
    track_ids: List[int]
    confidence: float = 1.0
    timestamp: float = 0.0
    frame_idx: int = 0


class DeepSortTracker:
    """
    DeepSORT追踪器封装
    将 deep-sort-realtime 的输出转换为统一的 TrackedObject 格式
    """

    def __init__(self, max_age=60, n_init=3, nn_budget=100,
                 embedder_gpu=True, half=True):
        """
        Args:
            max_age: 追踪最大存活帧数（目标消失后保留ID的帧数）
            n_init: 连续检测到n帧后才确认追踪
            nn_budget: 外观特征库大小
            embedder_gpu: 是否使用GPU提取外观特征
            half: 是否使用FP16加速
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget,
            embedder_gpu=embedder_gpu,
            half=half,
        )
        # 扩展信息存储：track_id -> DeepTrack
        self.tracks: Dict[int, DeepTrack] = {}

    def update(self, frame: np.ndarray,
               detections: List[Tuple[str, List[float], float]]) -> Dict[int, DeepTrack]:
        """
        更新追踪状态

        Args:
            frame: 当前帧图像(BGR)，DeepSORT需要提取外观特征
            detections: [(class_name, [x1,y1,x2,y2], confidence), ...]

        Returns:
            dict: {track_id: DeepTrack}
        """
        current_time = time.time()

        # 转换为 DeepSORT 输入格式: ([x,y,w,h], confidence, class_name)
        raw_detections = []
        for cls_name, bbox, conf in detections:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            raw_detections.append(([x1, y1, w, h], conf, cls_name))

        # DeepSORT更新
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)

        # 处理追踪结果
        active_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            active_ids.add(tid)

            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            bbox = [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            # 获取检测类别（从det_class获取）
            cls_name = track.det_class if hasattr(track, 'det_class') and track.det_class else 'unknown'
            conf = track.det_conf if hasattr(track, 'det_conf') and track.det_conf else 0.5

            if tid in self.tracks:
                # 更新已有追踪
                deep_track = self.tracks[tid]

                # 计算速度
                if deep_track.positions:
                    prev_cx, prev_cy = deep_track.positions[-1]
                    speed = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                    deep_track.speeds.append(speed)
                    if len(deep_track.speeds) > 30:
                        deep_track.speeds = deep_track.speeds[-30:]

                deep_track.bbox = bbox
                deep_track.confidence = conf
                deep_track.center = (cx, cy)
                deep_track.frame_count += 1
                deep_track.last_seen = current_time
                deep_track.positions.append((cx, cy))
                if len(deep_track.positions) > 60:
                    deep_track.positions = deep_track.positions[-60:]

                # 横向位置（用于变道检测）
                deep_track.lateral_positions.append(cx)
                if len(deep_track.lateral_positions) > 60:
                    deep_track.lateral_positions = deep_track.lateral_positions[-60:]

                # 停止判断
                avg_speed = np.mean(deep_track.speeds[-5:]) if deep_track.speeds else 0
                if avg_speed < 2.0:
                    deep_track.stopped_frames += 1
                else:
                    deep_track.stopped_frames = 0

                # 类别可能被DeepSORT更新
                if cls_name != 'unknown':
                    deep_track.class_name = cls_name

            else:
                # 新追踪
                self.tracks[tid] = DeepTrack(
                    track_id=tid,
                    class_name=cls_name,
                    bbox=bbox,
                    confidence=conf,
                    center=(cx, cy),
                    frame_count=1,
                    last_seen=current_time,
                    positions=[(cx, cy)],
                    speeds=[],
                    stopped_frames=0,
                    lateral_positions=[cx],
                )

        # 移除不活跃的追踪
        to_remove = []
        for tid in self.tracks:
            if tid not in active_ids:
                if current_time - self.tracks[tid].last_seen > 3.0:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

        # 只返回活跃的追踪
        return {tid: self.tracks[tid] for tid in self.tracks if tid in active_ids}


class DeepSortTrafficEventDetector:
    """
    基于DeepSORT的交通事件检测器

    相比SimpleTracker版本的优势:
    - 外观特征匹配: 遮挡后重新出现能保持ID
    - 追踪更稳定: 减少ID切换导致的事件误判
    - 支持紧急变道检测: 基于更稳定的轨迹
    """

    VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
    PERSON_CLASSES = {'person'}
    ALL_TRAFFIC_CLASSES = VEHICLE_CLASSES | PERSON_CLASSES

    def __init__(self, frame_width=1280, frame_height=720, fps=30,
                 max_age=None, n_init=3, embedder_gpu=True):
        """
        Args:
            frame_width: 视频帧宽度
            frame_height: 视频帧高度
            fps: 视频帧率
            max_age: 追踪最大存活帧数（默认=2*fps）
            n_init: 连续检测到n帧后才确认追踪
            embedder_gpu: 是否使用GPU提取外观特征
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.frame_idx = 0

        # DeepSORT追踪器
        self.tracker = DeepSortTracker(
            max_age=max_age or int(fps * 2),
            n_init=n_init,
            embedder_gpu=embedder_gpu,
        )

        # 事件配置
        self.config = {
            # 停车检测
            'stop_min_frames': int(fps * 3),
            'stop_speed_threshold': 2.0,

            # 逆行检测
            'wrong_way_min_frames': int(fps * 1.5),
            'wrong_way_movement_threshold': 30,

            # 拥堵检测
            'congestion_vehicle_count': 5,
            'congestion_area_ratio': 0.4,

            # 碰撞检测
            'collision_iou_threshold': 0.1,
            'collision_speed_threshold': 5.0,

            # 超速检测
            'speeding_threshold': 25.0,

            # 缓行检测
            'slow_speed_threshold': 5.0,
            'slow_min_frames': int(fps * 2),

            # 行人闯入
            'pedestrian_invasion_frames': int(fps * 1),

            # 紧急变道检测
            'lane_change_lateral_threshold': 50,   # 横向位移阈值（像素）
            'lane_change_min_frames': int(fps * 0.5),  # 变道最短时间
            'lane_change_max_frames': int(fps * 2),    # 变道最长时间

            # 事件冷却
            'event_cooldown': 5.0,
        }

        # 事件历史与冷却
        self.active_events: List[TrafficEvent] = []
        self.event_history: List[TrafficEvent] = []
        self.last_event_time: Dict[EventType, float] = {}
        # 按track_id+事件类型的冷却，避免同一目标重复触发
        self.last_event_by_target: Dict[Tuple[int, EventType], float] = {}

        # 区域定义
        self.road_region = (0.1, 0.4, 0.9, 0.95)

        # 方向定义
        self.normal_direction = 'right'

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

    def _get_displacement(self, positions, n=10):
        """获取最近n帧的位移向量"""
        if len(positions) < 2:
            return (0, 0)
        recent = positions[-min(n, len(positions)):]
        return (recent[-1][0] - recent[0][0], recent[-1][1] - recent[0][1])

    def _is_wrong_direction(self, dx, dy):
        """判断是否逆行"""
        t = self.config['wrong_way_movement_threshold']
        if self.normal_direction == 'right' and dx < -t:
            return True
        elif self.normal_direction == 'left' and dx > t:
            return True
        elif self.normal_direction == 'down' and dy < -t:
            return True
        elif self.normal_direction == 'up' and dy > t:
            return True
        return False

    def _compute_overlap_ratio(self, box1, box2):
        """计算重叠面积比"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / min(a1, a2) if min(a1, a2) > 0 else 0

    def _is_event_cooled(self, event_type: EventType, track_id: int = None) -> bool:
        """检查事件冷却（支持全局冷却和按目标冷却）"""
        # 全局冷却
        if event_type in self.last_event_time:
            if time.time() - self.last_event_time[event_type] < self.config['event_cooldown']:
                return False
        # 按目标冷却
        if track_id is not None:
            key = (track_id, event_type)
            if key in self.last_event_by_target:
                if time.time() - self.last_event_by_target[key] < self.config['event_cooldown'] * 2:
                    return False
        return True

    def _add_event(self, event: TrafficEvent, track_id: int = None):
        """添加事件"""
        if self._is_event_cooled(event.event_type, track_id):
            event.timestamp = time.time()
            event.frame_idx = self.frame_idx
            self.active_events.append(event)
            self.event_history.append(event)
            self.last_event_time[event.event_type] = event.timestamp
            if track_id is not None:
                self.last_event_by_target[(track_id, event.event_type)] = event.timestamp

    def _detect_stopped_vehicles(self, tracks: Dict[int, DeepTrack]):
        """检测停车/违停"""
        for tid, track in tracks.items():
            if track.class_name not in self.VEHICLE_CLASSES:
                continue
            if track.frame_count < self.config['stop_min_frames']:
                continue
            if track.stopped_frames >= self.config['stop_min_frames']:
                if self._is_in_road(track.center[0], track.center[1]):
                    self._add_event(TrafficEvent(
                        event_type=EventType.VEHICLE_STOPPED,
                        severity=SEVERITY_LEVELS[EventType.VEHICLE_STOPPED],
                        description=f"{track.class_name} ID:{tid} 停留{track.stopped_frames // self.fps:.0f}秒",
                        location=(track.center[0] / self.frame_width,
                                  track.center[1] / self.frame_height),
                        track_ids=[tid],
                        confidence=min(1.0, track.stopped_frames / (self.config['stop_min_frames'] * 2)),
                    ), track_id=tid)

    def _detect_wrong_way(self, tracks: Dict[int, DeepTrack]):
        """检测逆行"""
        for tid, track in tracks.items():
            if track.class_name not in self.VEHICLE_CLASSES:
                continue
            if len(track.positions) < self.config['wrong_way_min_frames']:
                continue
            dx, dy = self._get_displacement(track.positions)
            if self._is_wrong_direction(dx, dy):
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
                    ), track_id=tid)

    def _detect_pedestrian_intrusion(self, tracks: Dict[int, DeepTrack]):
        """检测行人闯入车道"""
        for tid, track in tracks.items():
            if track.class_name not in self.PERSON_CLASSES:
                continue
            if track.frame_count < self.config['pedestrian_invasion_frames']:
                continue
            if self._is_in_road(track.center[0], track.center[1]):
                nearby_vehicles = []
                for vid, vtrack in tracks.items():
                    if vtrack.class_name in self.VEHICLE_CLASSES:
                        dist = np.sqrt(
                            (track.center[0] - vtrack.center[0]) ** 2 +
                            (track.center[1] - vtrack.center[1]) ** 2
                        )
                        if dist < 200:
                            nearby_vehicles.append(vid)
                self._add_event(TrafficEvent(
                    event_type=EventType.PEDESTRIAN_INTRUSION,
                    severity=SEVERITY_LEVELS[EventType.PEDESTRIAN_INTRUSION],
                    description=f"行人 ID:{tid} 在车道内{'，附近有车辆' if nearby_vehicles else ''}",
                    location=(track.center[0] / self.frame_width,
                              track.center[1] / self.frame_height),
                    track_ids=[tid] + nearby_vehicles,
                    confidence=0.9 if nearby_vehicles else 0.6,
                ), track_id=tid)

    def _detect_congestion(self, tracks: Dict[int, DeepTrack]):
        """检测拥堵"""
        vehicles = [t for tid, t in tracks.items()
                    if t.class_name in self.VEHICLE_CLASSES and
                    self._is_in_road(t.center[0], t.center[1])]
        count = len(vehicles)
        if count >= self.config['congestion_vehicle_count']:
            avg_speeds = [np.mean(v.speeds[-10:]) for v in vehicles if v.speeds]
            mean_speed = np.mean(avg_speeds) if avg_speeds else 0
            if mean_speed < self.config['slow_speed_threshold']:
                self._add_event(TrafficEvent(
                    event_type=EventType.CONGESTION,
                    severity=SEVERITY_LEVELS[EventType.CONGESTION],
                    description=f"车道内{count}辆车，均速{mean_speed:.1f}，疑似拥堵",
                    location=(0.5, 0.5),
                    track_ids=[v.track_id for v in vehicles],
                    confidence=min(1.0, count / 10),
                ))

    def _detect_collision(self, tracks: Dict[int, DeepTrack]):
        """检测碰撞"""
        vehicle_tracks = {tid: t for tid, t in tracks.items()
                         if t.class_name in self.VEHICLE_CLASSES and len(t.speeds) >= 3}
        tids = list(vehicle_tracks.keys())
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                t1, t2 = vehicle_tracks[tids[i]], vehicle_tracks[tids[j]]
                overlap = self._compute_overlap_ratio(t1.bbox, t2.bbox)
                if overlap > self.config['collision_iou_threshold']:
                    s1 = np.mean(t1.speeds[-5:]) if t1.speeds else 0
                    s2 = np.mean(t2.speeds[-5:]) if t2.speeds else 0
                    s1r = np.mean(t1.speeds[-3:]) if len(t1.speeds) >= 3 else 0
                    s2r = np.mean(t2.speeds[-3:]) if len(t2.speeds) >= 3 else 0
                    if (s1 > self.config['collision_speed_threshold'] or
                            s2 > self.config['collision_speed_threshold']):
                        if s1r < s1 * 0.3 or s2r < s2 * 0.3 or \
                           t1.stopped_frames > 3 or t2.stopped_frames > 3:
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

    def _detect_speeding(self, tracks: Dict[int, DeepTrack]):
        """检测超速"""
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
                    description=f"{track.class_name} ID:{tid} 疑似超速({avg_speed:.1f})",
                    location=(track.center[0] / self.frame_width,
                              track.center[1] / self.frame_height),
                    track_ids=[tid],
                    confidence=min(1.0, avg_speed / (self.config['speeding_threshold'] * 2)),
                ), track_id=tid)

    def _detect_slow_moving(self, tracks: Dict[int, DeepTrack]):
        """检测缓行"""
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
                        description=f"{track.class_name} ID:{tid} 缓行({avg_speed:.1f})",
                        location=(track.center[0] / self.frame_width,
                                  track.center[1] / self.frame_height),
                        track_ids=[tid],
                        confidence=0.7,
                    ), track_id=tid)

    def _detect_sudden_lane_change(self, tracks: Dict[int, DeepTrack]):
        """
        检测紧急变道
        基于横向位移的突然变化判断
        """
        for tid, track in tracks.items():
            if track.class_name not in self.VEHICLE_CLASSES:
                continue
            if len(track.lateral_positions) < self.config['lane_change_min_frames']:
                continue

            lateral = track.lateral_positions
            min_frames = self.config['lane_change_min_frames']
            max_frames = self.config['lane_change_max_frames']
            threshold = self.config['lane_change_lateral_threshold']

            # 滑动窗口检测横向突变
            # 在 [min_frames, max_frames] 的窗口内，如果横向位移超过阈值
            for window in range(min_frames, min(max_frames + 1, len(lateral) + 1)):
                recent = lateral[-window:]
                lateral_shift = abs(recent[-1] - recent[0])

                if lateral_shift > threshold:
                    # 同时纵向位移较小（确认是变道而非转弯）
                    if len(track.positions) >= window:
                        recent_pos = track.positions[-window:]
                        dx = abs(recent_pos[-1][0] - recent_pos[0][0])
                        dy = abs(recent_pos[-1][1] - recent_pos[0][1])

                        # 横向位移 > 纵向位移 → 变道
                        if dx > dy * 0.5:
                            # 速度较快时变道更危险
                            avg_speed = np.mean(track.speeds[-5:]) if track.speeds else 0
                            self._add_event(TrafficEvent(
                                event_type=EventType.SUDDEN_LANE_CHANGE,
                                severity=SEVERITY_LEVELS[EventType.SUDDEN_LANE_CHANGE],
                                description=f"{track.class_name} ID:{tid} 疑似紧急变道(横向{lateral_shift:.0f}px)",
                                location=(track.center[0] / self.frame_width,
                                          track.center[1] / self.frame_height),
                                track_ids=[tid],
                                confidence=min(1.0, lateral_shift / (threshold * 3)),
                            ), track_id=tid)
                            break  # 一个目标每帧只报一次

    def detect(self, frame: np.ndarray,
               detections: List[Tuple[str, List[float], float]]) -> List[TrafficEvent]:
        """
        执行交通事件检测

        Args:
            frame: 当前帧图像(BGR)，DeepSORT需要提取外观特征
            detections: YOLO检测结果 [(class_name, [x1,y1,x2,y2], confidence), ...]

        Returns:
            当前帧检测到的事件列表
        """
        self.frame_idx += 1
        self.active_events = []

        # 过滤交通相关目标
        traffic_dets = [d for d in detections if d[0] in self.ALL_TRAFFIC_CLASSES]

        if not traffic_dets:
            return self.active_events

        # 更新DeepSORT追踪器
        tracks = self.tracker.update(frame, traffic_dets)

        if not tracks:
            return self.active_events

        # 执行各项事件检测
        self._detect_stopped_vehicles(tracks)
        self._detect_wrong_way(tracks)
        self._detect_pedestrian_intrusion(tracks)
        self._detect_congestion(tracks)
        self._detect_collision(tracks)
        self._detect_speeding(tracks)
        self._detect_slow_moving(tracks)
        self._detect_sudden_lane_change(tracks)

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
