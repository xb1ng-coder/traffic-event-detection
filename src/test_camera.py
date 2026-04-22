"""
YOLOv8实时摄像头检测
确保虚拟环境已激活：source venv/bin/activate
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

print("🚀 交通事件实时检测系统 - YOLOv8")
print("=" * 50)


class RealTimeTrafficDetector:
    def __init__(self, model_path='models/yolov8n.pt', camera_index=0):
        """
        初始化实时检测器
        
        Args:
            model_path: YOLOv8模型路径
            camera_index: 摄像头索引（默认0为电脑内置摄像头）
        """
        self.camera_index = camera_index
        self.cap = None
        self.model = None
        self.is_running = False
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0
        
        # 初始化摄像头
        self.init_camera()
        
        # 加载模型
        self.load_model(model_path)
        
        # 检测统计
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'class_counts': {},
            'fps_history': []
        }
    
    def init_camera(self):
        """初始化摄像头"""
        print("\n📹 初始化摄像头...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"❌ 无法打开摄像头 {self.camera_index}")
            print("   请检查：")
            print("   1. 摄像头是否正确连接")
            print("   2. 摄像头权限是否开启")
            print("   3. 尝试其他摄像头索引（0, 1, 2...）")
            return False
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"   ✓ 摄像头 {self.camera_index} 已连接")
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps:.1f} FPS")
        
        return True
    
    def load_model(self, model_path):
        """加载YOLOv8模型"""
        print("\n🤖 加载YOLOv8模型...")
        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"   ✓ 从本地加载模型: {model_path}")
            else:
                print("   ⏳ 下载YOLOv8n模型（首次运行需要下载）...")
                self.model = YOLO('yolov8n.pt')
                os.makedirs('models', exist_ok=True)
                self.model.save(model_path)
                print(f"   ✓ 模型已保存: {model_path}")
        except Exception as e:
            print(f"   ❌ 模型加载失败: {e}")
            return False
        
        # 预热模型
        print("   🔥 模型预热...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.model(dummy_frame, verbose=False)
        print("   ✓ 模型准备就绪")
        
        return True
    
    def draw_detections(self, frame, results):
        """在帧上绘制检测结果"""
        # 获取检测结果
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                
                # 更新统计
                self.stats['class_counts'][cls_name] = self.stats['class_counts'].get(cls_name, 0) + 1
                
                # 绘制边界框
                x1, y1, x2, y2 = map(int, bbox)
                color = self.get_color_for_class(cls_name)
                
                # 绘制矩形
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签背景
                label = f"{cls_name} {confidence:.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                cv2.rectangle(frame, (x1, y1 - label_height - 10), 
                            (x1 + label_width, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def get_color_for_class(self, class_name):
        """根据类别返回颜色"""
        color_map = {
            'person': (0, 255, 0),       # 绿色
            'car': (255, 0, 0),         # 蓝色
            'truck': (0, 255, 255),     # 黄色
            'bus': (255, 255, 0),       # 青色
            'motorcycle': (255, 0, 255), # 紫色
            'bicycle': (0, 165, 255),   # 橙色
        }
        return color_map.get(class_name, (255, 255, 255))  # 默认白色
    
    def draw_stats(self, frame):
        """在帧上绘制统计信息"""
        height, width = frame.shape[:2]
        
        # 绘制统计信息面板
        stats_bg = np.zeros((200, 300, 3), dtype=np.uint8)
        
        # 计算面板位置
        panel_x, panel_y = 10, 10
        
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + 300, panel_y + 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制文本
        y_offset = 40
        cv2.putText(frame, "🚦 实时交通检测", (panel_x + 10, panel_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (panel_x + 10, panel_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(frame, f"帧数: {self.stats['total_frames']}", (panel_x + 10, panel_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        cv2.putText(frame, f"检测数: {self.stats['total_detections']}", (panel_x + 10, panel_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        
        # 显示类别统计
        cv2.putText(frame, "📊 类别统计:", (panel_x + 10, panel_y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 1)
        y_offset += 25
        
        # 显示最多5个类别
        sorted_classes = sorted(self.stats['class_counts'].items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        
        for i, (cls_name, count) in enumerate(sorted_classes):
            color = self.get_color_for_class(cls_name)
            text = f"  {cls_name}: {count}"
            cv2.putText(frame, text, (panel_x + 10, panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
        
        return frame
    
    def analyze_traffic_events(self, class_counts):
        """分析交通事件"""
        events = []
        
        # 车辆计数
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        vehicle_count = sum(class_counts.get(cls, 0) for cls in vehicle_classes)
        
        if vehicle_count >= 5:
            events.append("⚠️ 交通拥堵")
        elif vehicle_count >= 3:
            events.append("🚗 车流密集")
        
        # 行人检测
        if 'person' in class_counts and class_counts['person'] > 0:
            events.append(f"👤 行人{class_counts['person']}人")
        
        return events
    
    def draw_events(self, frame, events):
        """在帧上绘制事件警告"""
        if not events:
            return frame
        
        height, width = frame.shape[:2]
        events_panel_x, events_panel_y = width - 310, 10
        
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (events_panel_x, events_panel_y), 
                     (events_panel_x + 300, events_panel_y + 50 + len(events) * 25), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制事件标题
        cv2.putText(frame, "🚨 交通事件:", (events_panel_x + 10, events_panel_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 绘制事件列表
        y_offset = 55
        for i, event in enumerate(events):
            color = (0, 0, 255) if "⚠️" in event else (0, 255, 255)  # 红色警告，黄色提示
            cv2.putText(frame, f"  {event}", (events_panel_x + 10, events_panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 25
        
        return frame
    
    def calculate_fps(self):
        """计算FPS"""
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time >= 1.0:  # 每秒更新一次
            self.fps = self.frame_count / elapsed_time
            self.stats['fps_history'].append(self.fps)
            
            # 保持最近10个FPS值
            if len(self.stats['fps_history']) > 10:
                self.stats['fps_history'] = self.stats['fps_history'][-10:]
            
            # 重置计数
            self.frame_count = 0
            self.start_time = current_time
        
        return self.fps
    
    def run(self, confidence_threshold=0.3):
        """运行实时检测"""
        if not self.cap or not self.model:
            print("❌ 摄像头或模型初始化失败")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        print("\n▶️ 开始实时检测...")
        print("   按 'q' 键退出")
        print("   按 's' 键保存当前帧")
        print("   按 'p' 键暂停/继续")
        
        paused = False
        
        while self.is_running:
            if not paused:
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 无法读取帧")
                    break
                
                # 更新帧计数
                self.stats['total_frames'] += 1
                
                # 计算FPS
                self.calculate_fps()
                
                # 运行检测
                results = self.model(frame, conf=confidence_threshold, verbose=False)
                
                # 更新检测统计
                if results[0].boxes is not None:
                    detection_count = len(results[0].boxes)
                    self.stats['total_detections'] += detection_count
                
                # 绘制检测结果
                frame = self.draw_detections(frame, results)
                
                # 分析交通事件
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    class_counts = {}
                    boxes = results[0].boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = results[0].names[cls_id]
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    
                    events = self.analyze_traffic_events(class_counts)
                    frame = self.draw_events(frame, events)
                
                # 绘制统计信息
                frame = self.draw_stats(frame)
                
                # 显示帧
                cv2.imshow('YOLOv8 实时交通检测', frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # 退出
                print("\n⏹️ 停止检测...")
                self.is_running = False
                break
            elif key == ord('s'):  # 保存当前帧
                if not paused:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"results/capture_{timestamp}.jpg"
                    os.makedirs("results", exist_ok=True)
                    cv2.imwrite(filename, frame)
                    print(f"   📸 已保存截图: {filename}")
            elif key == ord('p'):  # 暂停/继续
                paused = not paused
                status = "暂停" if paused else "继续"
                print(f"   ⏸️  {status}检测")
        
        # 释放资源
        self.release()
    
    def release(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print(f"✅ 检测完成")
        print(f"   总帧数: {self.stats['total_frames']}")
        print(f"   总检测数: {self.stats['total_detections']}")
        print(f"   平均FPS: {np.mean(self.stats['fps_history']):.1f}" if self.stats['fps_history'] else "   平均FPS: N/A")


def main():
    """主函数"""
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 模型路径
    model_path = 'models/yolov8n.pt'
    
    # 创建检测器实例
    detector = RealTimeTrafficDetector(model_path=model_path, camera_index=0)
    
    # 运行检测
    detector.run(confidence_threshold=0.3)


if __name__ == "__main__":
    main()