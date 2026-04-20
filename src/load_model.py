"""
    这是一个加载模型的脚本，主要用于测试模型是否能够正确加载，并且在首次运行时下载模型文件。
"""
from ultralytics import YOLO

def main():
    print("\n 加载模型...")
    try:
        # 
        model = YOLO('models/yolov5n.pt')
        print("   ✓ 从本地加载模型")
    except:
        print("   ⏳ 下载YOLOv5n模型（首次运行需要下载）...")
        model = YOLO('yolov5n.pt')
        # 保存模型以便下次使用
        import os
        os.makedirs('models', exist_ok=True)
        model.save('models/yolov5n.pt')
        print("   ✓ 模型已保存: models/yolov5n.pt")

if __name__ == "__main__":
    main()