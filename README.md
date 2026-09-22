# Barcode Detection

基于 YOLOv8n 的 1D 条形码检测模型（单类：barcode），只定位条形码在图像中的位置，不识别具体数字。

## v2 精度升级（针对实拍"误检多/漏检"反馈）

| 项目 | 旧版 v3（FAST） | 新版 v2（高精度） |
|---|---|---|
| 权重初始化 | `yolov8n.yaml` 从头训练 | **本地预训练 `weights/yolov8n.pt` 迁移学习** |
| 数据用量 | `fraction=0.2`（仅 20%） | `fraction=1.0`（全量） |
| 输入尺寸 | 416 | **640**（细长/小条码不漏检） |
| 训练轮数 | 20 | 100 + 早停（patience=20） |
| 旋转增强 | 无（degrees=0） | **degrees=180**（治 180° 倒置漏检） |
| 负样本 | 纯 QR/无标签图全删除 | **保留为空标签负样本**（治成分表/文字误检） |
| 透视/亮度 | 默认 | 开启（治反光/倾斜/曲面） |

## 基线指标（v3 FAST）

Test 集（312 张）：P=0.919，R=0.839，mAP50=0.860，mAP50-95=0.469，best.pt 5.93 MB。
v2 目标：在实拍难例上 **Precision↑（降误检）、Recall↑（降漏检）**。

## 环境要求

- Python 3.10+
- PyTorch 2.0+（GPU 训练建议 CUDA）
- ultralytics >= 8.1，kagglehub

```bash
pip install ultralytics kagglehub
```

## 快速开始

### 1. 准备数据集（含负样本）
```bash
python scripts/prepare_dataset.py
```
自动下载 [kushagrapandya/barcode-detection](https://www.kaggle.com/datasets/kushagrapandya/barcode-detection)，
barcode 保留为正样本，**纯 QR / 无标签图保留为空标签负样本**，让模型学会"QR/文字不是 1D 条码"。

### 2. 训练（高精度）
```bash
python scripts/train.py
```
- 预训练权重已随仓库放在 `weights/yolov8n.pt`，**无需联网下载**。
- 8GB 显存若 OOM，把 `batch=16` 改成 `8` 或 `4`。
- 快速冒烟验证管线：`QUICK=1 python scripts/train.py`（10% 数据 / 15 epoch）。

### 3. 实拍难例回归评估
```bash
python scripts/eval_real.py
```
对 `real_eval/` 里 21 张实拍难例（反光/倒置/倾斜/多码/模糊低光）出报告，
自动标出【低置信】和【多框可疑】的图——这些就是下一轮要人工标注并加进训练集的难例。

### 4. 推理
```bash
python scripts/predict.py
```
或在代码中：
```python
from ultralytics import YOLO
model = YOLO("runs/barcode_hifi_v2/weights/best.pt")
results = model.predict(source="path/to/images", conf=0.30, save=True)
```

## 项目结构

```
barcode_detect/
├── weights/
│   └── yolov8n.pt            # 预训练权重（随仓库提供，离线可用）
├── scripts/
│   ├── prepare_dataset.py    # 数据下载 + 清洗 + 负样本保留
│   ├── train.py              # 高精度训练（迁移学习 + 旋转/透视/亮度增强）
│   ├── eval_real.py          # 实拍难例回归评估
│   └── predict.py            # 推理 + 可视化
├── real_eval/                # 21 张实拍难例回归测试集
├── runs/                     # 训练输出（不入库）
├── data/                     # 数据集（不入库）
└── .gitignore
```

## 注意事项

- **GitHub 不可达**：`train.py` 内置了离线补丁，且预训练权重已本地提供，无需联网。
- **Windows 多进程**：`workers=0` 避免 DataLoader pickle 崩溃；Linux/WSL 可改 `workers=8`。
- **训练结果路径**：高精度权重在 `runs/barcode_hifi_v2/weights/best.pt`；`eval_real.py` 会自动优先找它。

## 后续：继续扩数据集（老师建议）

当前只用了 Kaggle 一个数据源，下一步可接入：
- **BarBeR**（GitHub: `Henvezz95/BarBeR`，8748 图 / 多码同帧 / 遮挡模糊）
- **YOLO-Barcode**（`ftp://smartengines.com/yolo-barcode`，小条码 / 透视变形）
- **SBD 合成集**（GitHub: `viplabB/SBD`，10 万合成图，可造反光/曲面难例负样本）
- 把 `eval_real.py` 标出的难例人工标注后并入训练集。
