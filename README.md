# 交通事件检测系统 - 毕业设计

面向边缘计算节点的轻量化交通事件提取方法及部署研究

## 项目结构
- `src/` - 源代码
- `data/` - 数据集
- `models/` - 模型文件
- `notebooks/` - 实验笔记
- `web/` - Web界面
- `results/` - 实验结果

## 环境配置
1. 创建虚拟环境：`python3 -m venv venv`
2. 激活环境：`source venv/bin/activate`
3. 安装依赖：`pip install -r requirements.txt`

## 使用方法
1. 训练模型：`python src/train.py`
2. 运行检测：`python src/detect.py`
3. 启动Web：`python web/app.py`
