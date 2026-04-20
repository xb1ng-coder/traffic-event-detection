"""
YOLOv8测试脚本
确保虚拟环境已激活：source venv/bin/activate
"""
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import glob

print("🚀 交通事件检测系统 - YOLOv8测试")
print("=" * 50)


def load_images_from_folder(folder="data/images"):
    """从文件夹加载所有图片"""
    supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
    image_paths = []
    for fmt in supported_formats:
        image_paths.extend(glob.glob(os.path.join(folder, fmt)))
    return sorted(image_paths)


def detect_and_analyze(model, image_path):
    """对单张图片运行检测并分析结果"""
    print(f"\n📸 正在处理: {os.path.basename(image_path)}")
    print("-" * 40)

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"   ❌ 无法读取图片: {image_path}")
        return

    print(f"   图片尺寸: {img.shape[1]}x{img.shape[0]}")

    # 运行检测
    results = model(img, conf=0.3, verbose=False)

    # 保存结果图片
    result_img = results[0].plot()
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"results/{filename}_detected.jpg"
    os.makedirs("results", exist_ok=True)
    cv2.imwrite(output_path, result_img)
    print(f"   ✓ 结果已保存: {output_path}")

    # 显示图片
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(result_img_rgb)
    plt.title(f"Detection Result - {os.path.basename(image_path)}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plot_path = f"results/{filename}_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    # 关闭窗口继续检测
    # plt.show()
    # 直接保存继续检测
    plt.close()

    # 打印检测统计
    print("\n📊 检测统计:")
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        print(f"   检测到目标总数: {len(boxes)}")

        class_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            confidence = float(box.conf[0])
            bbox = [round(x, 1) for x in box.xyxy[0].tolist()]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            print(f"   - {cls_name}: 置信度 {confidence:.2f}, 位置 {bbox}")

        print(f"\n   类别统计:")
        for cls_name, count in class_counts.items():
            print(f"     {cls_name}: {count} 个")

        # 交通事件分析
        print(f"\n🚦 交通事件分析:")
        vehicle_count = sum(count for name, count in class_counts.items()
                            if name in ['car', 'truck', 'bus', 'motorcycle'])
        if vehicle_count >= 3:
            print(f"   ⚠️  检测到 {vehicle_count} 辆车，可能存在拥堵")
        if 'person' in class_counts:
            print(f"   👤 检测到 {class_counts['person']} 名行人，注意安全")
        if vehicle_count == 0 and 'person' not in class_counts:
            print(f"   ✅ 未发现明显交通事件")
    else:
        print("   未检测到任何目标")


def main():
    # 1. 扫描图片
    print("\n1. 扫描 data/images 目录...")
    image_paths = load_images_from_folder("data/images")

    if not image_paths:
        print("   ❌ 未找到任何图片，请将图片放入 data/images/ 目录")
        print("   支持格式: jpg, jpeg, png, bmp, webp")
        return

    print(f"   ✓ 找到 {len(image_paths)} 张图片:")
    for path in image_paths:
        print(f"     - {os.path.basename(path)}")

    # 2. 加载模型
    print("\n2. 加载YOLOv8模型...")
    try:
        model = YOLO('models/yolov8n.pt')
        print("   ✓ 从本地加载模型")
    except:
        print("   ⏳ 下载YOLOv8n模型（首次运行需要下载）...")
        model = YOLO('yolov8n.pt')
        os.makedirs('models', exist_ok=True)
        model.save('models/yolov8n.pt')
        print("   ✓ 模型已保存: models/yolov8n.pt")

    # 3. 逐张处理图片
    print(f"\n3. 开始检测（共 {len(image_paths)} 张）...")
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}]", end="")
        detect_and_analyze(model, image_path)

    print("\n" + "=" * 50)
    print(f"✅ 全部完成！结果保存在 results/ 目录")


if __name__ == "__main__":
    main()