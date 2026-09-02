# Barcode Detection

基于 YOLOv8n 的 1D 条形码检测模型（单类：barcode），只定位条形码在图像中的位置，不识别具体数字。

## 模型性能

Test 集（312 张）评估结果：

| Metric | Value |
|--------|-------|
| Precision | 0.919 |
| Recall | 0.839 |
| mAP@50 | 0.860 |
| mAP@50-95 | 0.469 |

模型体积：**best.pt 5.93 MB**

## 环境要求

- Python 3.10+
- PyTorch 2.0+ (with CUDA for GPU training)
- ultralytics >= 8.1

```bash
pip install ultralytics
```

## 快速开始

### 1. 准备数据集

使用 [kushagrapandya/barcode-detection](https://www.kaggle.com/datasets/kushagrapandya/barcode-detection) 数据集（自动下载 + 过滤 QR Code 类别）：

```bash
python scripts/prepare_dataset.py
```

### 2. 训练

默认 FAST 模式配置（20% 数据、imgsz=416、batch=16、AMP），适合快速迭代：

```bash
python scripts/train.py
```

训练配置：

| Param | Value |
|-------|-------|
| model | yolov8n.yaml |
| imgsz | 416 |
| batch | 16 |
| fraction | 0.2 |
| epochs | 20 |
| optimizer | SGD |
| amp | True |

训练完成后自动在 test 集上评估，权重保存在 `runs/barcode_fast_v3/weights/best.pt`。

### 3. 推理

```bash
python scripts/predict.py
```

结果保存在 `runs/predictions/barcode_test_results/`，包含带检测框的可视化图片和坐标标签。

也可以对自己的图片/文件夹推理：

```python
from ultralytics import YOLO

model = YOLO("runs/barcode_fast_v3/weights/best.pt")
results = model.predict(source="path/to/images", conf=0.25, save=True)
```

## 项目结构

```
barcode_detect/
├── scripts/
│   ├── prepare_dataset.py   # 数据集下载 + QR Code 过滤
│   ├── train.py             # 训练脚本（含 GitHub 下载绕过 patch）
│   └── predict.py           # 推理 + 可视化
├── runs/                    # 训练输出（不在版本控制中）
│   └── barcode_fast_v3/
│       └── weights/
│           ├── best.pt      # 最佳权重
│           └── last.pt
├── data/                    # 数据集（不在版本控制中）
└── .gitignore
```

## 注意事项

- **GitHub 不可达环境**：本项目的脚本内置了 monkey-patch，会自动跳过 ultralytics 内部对 GitHub/ultralytics.com 的权重下载请求。如果你的环境可以正常访问 GitHub，删除 `train.py` 顶部的 patch 代码块即可。
- **Windows 多进程**：`workers=0` 是为了避免 Windows 下 DataLoader 多进程 pickle 崩溃。如果在 Linux/WSL 上运行，可以改为 `workers=8` 加速训练。
- **fast 模式精度有限**：使用了 20% 数据 + imgsz=416 做快速迭代。生产环境建议用全量数据 + imgsz=640 + epochs=50 重新训练。
