# Barcode Detection

基于 YOLOv8n 的 1D 条形码检测模型（单类：barcode），只定位条形码在图像中的位置，不识别具体数字。

## 精度升级历程

针对实拍场景「误检多、漏检多」的反馈，做了两轮升级：

| 项目 | 旧版 v3（FAST） | **新版 v4（当前）** |
|---|---|---|
| 权重初始化 | `yolov8n.yaml` 从头训练 | **本地预训练 `weights/yolov8n.pt` 迁移学习** |
| 数据用量 | `fraction=0.2`（仅 20%） | `fraction=1.0`（全量 28696 张） |
| 输入尺寸 | 416 | **640** |
| 训练轮数 | 20 | 100（patience=20，实际跑满 100 轮） |
| 旋转增强 | 无（degrees=0） | **degrees=180**（治 180° 倒置） |
| 负样本 | 纯 QR / 无标签图全删除 | **保留为空标签负样本**（治成分表/文字误检） |
| 透视/亮度 | 默认 | 开启（治反光、倾斜、曲面） |
| DataLoader | — | `workers=4` |

## 最终指标（v4，跑满 100 轮）

### 测试集（432 张 = 312 正样本 + 120 负样本）

| 指标 | 旧版 v3 | **新版 v4** | 变化 |
|---|---|---|---|
| Precision | 0.919 | **0.929** | +1.1% |
| Recall | 0.839 | **0.878** | **+4.6%** |
| mAP50 | 0.860 | **0.914** | **+6.2%** |
| mAP50-95 | 0.469 | 0.472 | +0.6% |

验证集（2382 张）：P=0.968，R=0.948，mAP50=0.966，mAP50-95=0.653。

### 实拍难例（19 张，conf=0.25）

| 指标 | 旧版 v3 | **新版 v4** |
|---|---|---|
| 检出率 | 18/19 (94.7%) | 18/19 (94.7%) |
| 平均置信度 | 0.689 | **0.698** |
| **多余框（误检）** | 9 | **6（↓33%）** |

成效最明显的两张：

| 图片 | v3 | v4 |
|---|---|---|
| `05-plastic-bag-crumpled`（成分表文字误检） | 5 框 | **1 框 / 0.73** |
| `10-rotated-90-paper-sheet`（质检文字误检） | 5 框 | **4 框** |

## ⚠️ 已知未解决问题

**`01-upside-down-180-pen-label.JPG`（180° 倒置的细长条码）仍为 0 框漏检。**

训练已开 `degrees=180`，但「倒置 + 极细长 + 圆柱笔身 + 反光」的组合仍未学会。仅靠数据增强无法解决，**需要人工补充该类型难例图片进训练集**。

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
barcode 保留为正样本，**纯 QR / 无标签图保留为空标签负样本**，让模型学会「QR/文字不是 1D 条码」。

### 2. 训练
```bash
python scripts/train.py
```
- 预训练权重已随仓库放在 `weights/yolov8n.pt`，**无需联网下载**。
- 支持环境变量：`DATA`（data.yaml 路径）、`NAME`（run 名）、`EPOCHS`、`BATCH`、`WORKERS`、`QUICK`。
- 8GB 显存若 OOM，把 `batch=16` 改成 `8` 或 `4`。
- 快速冒烟验证管线：`QUICK=1 python scripts/train.py`。

### 3. 断点续训
```bash
python scripts/resume_train.py
# 或指定检查点
python scripts/resume_train.py runs/xxx/weights/last.pt
```

### 4. 实拍难例回归评估
```bash
python scripts/eval_real.py
```
对 `real_eval/` 里 21 张实拍难例出报告，自动标出【低置信】和【多框可疑】的图。

### 5. 新旧模型并排对比
```bash
python scripts/compare_models.py
```
支持环境变量：`W_OLD`、`W_NEW`、`REAL_DIR`、`OUT_DIR`、`CONF`、`MAX_SIDE`。

### 6. 推理
```bash
python scripts/predict.py
```
或在代码中：
```python
from ultralytics import YOLO
model = YOLO("runs/barcode_hifi_v4/best.pt")
results = model.predict(source="path/to/images", conf=0.45, save=True)
```

> **提示**：实拍场景建议把 `conf` 提到 **0.45~0.5**。剩余的多余框大多置信在 0.30 附近，
> 直接调阈值即可零成本压掉一批误检。

## 数据集分析工具

| 脚本 | 用途 |
|---|---|
| `analyze_dataset.py` | 数据集冗余度分析（按 `.rf.<hash>` 后缀归组，可发现离线增强造成的重复） |
| `prepare_dataset_dedup.py` | 用硬链接建去重数据集（注意：会同时砍掉负样本，实测**效果更差**） |
| `count_negatives.py` | 统计正/负样本数量 |
| `check_image_sizes.py` | 图片尺寸分布统计 |

> **踩坑记录**：去重方案（`data/barcode_dedup`）会把负样本也砍掉约 4 倍（3225 → 806），
> 而负样本正是压制误检的核心，实测多余框反而从 9 升到 22。**不要用去重数据集训练。**

## 项目结构

```
barcode_detect/
├── weights/
│   └── yolov8n.pt              # 预训练权重（随仓库提供，离线可用）
├── scripts/
│   ├── prepare_dataset.py      # 数据下载 + 清洗 + 负样本保留
│   ├── train.py                # 训练（迁移学习 + 旋转/透视/亮度增强）
│   ├── resume_train.py         # 断点续训
│   ├── eval_real.py            # 实拍难例回归评估
│   ├── compare_models.py       # 新旧模型并排对比
│   ├── predict.py              # 推理 + 可视化
│   ├── analyze_dataset.py      # 数据集冗余度分析
│   ├── prepare_dataset_dedup.py# 硬链接去重建集（效果差，仅作记录）
│   ├── count_negatives.py      # 正/负样本计数
│   └── check_image_sizes.py    # 图片尺寸分布
├── runs/barcode_hifi_v4/
│   ├── best.pt                 # v4 训练好的权重（6.0 MB）
│   ├── results.csv             # 100 轮完整训练曲线
│   └── args.yaml               # 训练超参
├── real_eval/                  # 21 张实拍难例回归测试集
└── .gitignore
```

## 注意事项

- **GitHub 不可达**：`train.py` 内置离线补丁，预训练权重已本地提供，无需联网。
- **Windows 多进程**：`workers=4` 经实测可稳定跑完 100 轮；`workers=8` 会在第 5 轮附近死锁（worker 变僵尸并占住显存），**不要用 8**。
- **训练结果路径**：v4 权重在 `runs/barcode_hifi_v4/best.pt`。

## 后续：继续扩数据集

当前只用了 Kaggle 一个数据源，下一步可接入：

- **BarBeR**（GitHub: `Henvezz95/BarBeR`，8748 图 / 多码同帧 / 遮挡模糊）
- **YOLO-Barcode**（`ftp://smartengines.com/yolo-barcode`，小条码 / 透视变形）
- **SBD 合成集**（GitHub: `viplabB/SBD`，10 万合成图，可造反光/曲面难例负样本）
- **优先补「倒置 + 细长条码」难例**——这是目前唯一未解决的漏检类型。

另外 v4 的最佳轮次是最后一轮（ep100）且指标仍在上升，说明 100 轮没跑够，
可以把 `EPOCHS` 提到 150~200 再训一轮。
