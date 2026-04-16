"""
YOLOv8测试脚本
确保虚拟环境已激活：source venv/bin/activate
"""
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

print("🚀 交通事件检测系统 - YOLOv8测试")
print("=" * 50)

def create_sample_traffic_image(width=800, height=600):
    """创建示例交通图像"""
    # 创建黑色背景
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 绘制蓝天背景
    img[:, :, 0] = 135  # 蓝色通道
    img[:, :, 1] = 206  # 绿色通道
    img[:, :, 2] = 235  # 红色通道
    
    # 绘制道路
    road_top = height // 2
    cv2.rectangle(img, (0, road_top), (width, height), (50, 50, 50), -1)
    
    # 绘制车道线
    for x in range(0, width, 80):
        cv2.rectangle(img, (x, road_top + 100), (x+40, road_top + 110), (255, 255, 0), -1)
    
    # 绘制车辆
    # 红色小汽车
    cv2.rectangle(img, (100, road_top + 50), (250, road_top + 150), (0, 0, 255), -1)
    cv2.rectangle(img, (120, road_top + 30), (230, road_top + 50), (0, 0, 200), -1)  # 车顶
    
    # 蓝色卡车
    cv2.rectangle(img, (400, road_top + 30), (600, road_top + 180), (255, 0, 0), -1)
    cv2.rectangle(img, (450, road_top - 20), (550, road_top + 30), (200, 0, 0), -1)  # 车头
    
    # 绿色公交车
    cv2.rectangle(img, (650, road_top + 20), (780, road_top + 160), (0, 255, 0), -1)
    
    # 添加文字
    cv2.putText(img, "Traffic Scene Simulation", (200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(img, "For YOLOv8 Testing", (250, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
    
    return img

def main():
    print("1. 创建示例交通场景...")
    test_img = create_sample_traffic_image()
    cv2.imwrite("data/images/test_scene.jpg", test_img)
    print("   ✓ 图片已保存: data/images/test_scene.jpg")
    
    print("\n2. 加载YOLOv8模型...")
    try:
        # 尝试加载本地模型
        model = YOLO('models/yolov8n.pt')
        print("   ✓ 从本地加载模型")
    except:
        print("   ⏳ 下载YOLOv8n模型（首次运行需要下载）...")
        model = YOLO('yolov8n.pt')
        # 保存模型以便下次使用
        import os
        os.makedirs('models', exist_ok=True)
        model.save('models/yolov8n.pt')
        print("   ✓ 模型已保存: models/yolov8n.pt")
    
    print("\n3. 运行目标检测...")
    # 运行检测
    results = model(test_img, conf=0.3, verbose=False)
    
    print("\n4. 显示检测结果...")
    # 获取带标注的图像
    result_img = results[0].plot()
    
    # 保存结果
    cv2.imwrite("results/first_detection.jpg", result_img)
    print("   ✓ 结果已保存: results/first_detection.jpg")
    
    # 显示图片
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(result_img_rgb)
    plt.title("YOLOv8 交通场景检测演示", fontsize=16, fontproperties='SimHei')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('results/detection_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印检测统计
    print("\n" + "=" * 50)
    print("📊 检测统计:")
    print("-" * 30)
    
    if results[0].boxes is not None:
        boxes = results[0].boxes
        print(f"检测到目标总数: {len(boxes)}")
        
        # 统计各类别
        class_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            print(f"  - {cls_name}: 置信度 {confidence:.2f}, 位置 {bbox}")
        
        print(f"\n类别统计:")
        for cls_name, count in class_counts.items():
            print(f"  {cls_name}: {count}个")
        
        # 分析可能的交通事件
        print(f"\n🚦 交通事件分析:")
        vehicle_count = sum(count for name, count in class_counts.items() 
                          if name in ['car', 'truck', 'bus', 'motorcycle'])
        
        if vehicle_count >= 3:
            print(f"  ⚠️  检测到 {vehicle_count} 辆车，可能存在拥堵")
        if 'person' in class_counts:
            print(f"  👤 检测到行人，注意安全")
    else:
        print("未检测到目标")
    
    print("\n✅ 测试完成！")
    print("下一步: 尝试用真实交通图片或视频进行测试")

if __name__ == "__main__":
    main()
