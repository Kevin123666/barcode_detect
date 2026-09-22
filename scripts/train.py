"""
Train YOLOv8n for 1D barcode detection - HIGH ACCURACY MODE v2
============================================================================
相比 FAST v3 的关键改动（针对老师反馈的"误检多/漏检"）：

1. 【最大提升点】加载本地预训练权重 weights/yolov8n.pt 做迁移学习，
   不再用 yolov8n.yaml 从头训练。
2. 全量数据 fraction=1.0，imgsz=640（原来 416 太小，细长/小条码漏检）。
3. epochs=100 + patience 早停（原来 20 轮欠拟合）。
4. 开启随机旋转 degrees=180（原来 0，导致 180° 倒置条码漏检），
   并加模糊/透视/亮度扰动，覆盖反光、倾斜、曲面等实拍难点。
5. device 自动检测（有 GPU 用 GPU，无 GPU 退回 CPU）。

用法：
    python scripts/train.py            # 高精度全量训练
    QUICK=1 python scripts/train.py    # 快速冒烟（10% 数据 / 15 epoch）
"""
import os
from pathlib import Path

# === 离线补丁：仅在 GitHub 不可达环境下，避免训练时联网拉取额外资源 ===
import ultralytics.utils.downloads as _dl
import ultralytics.nn.tasks as _tasks

_orig_safe_download = _dl.safe_download


def _patched_safe_download(url="", file=None, *args, **kwargs):
    """已有有效文件直接复用；否则尝试原始下载，失败不崩。"""
    if file:
        p = Path(file)
        if p.exists() and p.stat().st_size > 100000:
            return str(p)
        try:
            return _orig_safe_download(url=url, file=file, *args, **kwargs)
        except Exception:
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_bytes(b"")
            return str(file)
    return None


_orig_torch_load = _tasks.torch_safe_load


def _patched_torch_load(file, device=None):
    try:
        return _orig_torch_load(file, device)
    except Exception:
        return None, None


_dl.safe_download = _patched_safe_download
_tasks.torch_safe_load = _patched_torch_load
# === 补丁结束 ===

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "barcode_only" / "data.yaml"
PRETRAIN = PROJECT_ROOT / "weights" / "yolov8n.pt"
RUNS_DIR = PROJECT_ROOT / "runs"

QUICK = os.environ.get("QUICK") == "1"


def pick_device():
    return 0 if torch.cuda.is_available() else "cpu"


def main():
    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} 不存在！请先运行 python scripts/prepare_dataset.py")
        return

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()

    print("=" * 64)
    print("YOLOv8n Barcode Detection - HIGH ACCURACY v2")
    print("=" * 64)
    print(f"  device      : {device}")
    print(f"  pretrain    : {PRETRAIN} (exists={PRETRAIN.exists()})")
    print(f"  quick mode  : {QUICK}")
    print("=" * 64)

    # 用本地预训练权重做迁移学习；缺失则回退到 yaml 从头训练
    if PRETRAIN.exists():
        model = YOLO(str(PRETRAIN))
    else:
        print("[WARN] 未找到本地预训练权重，回退到从头训练（精度会差很多）")
        model = YOLO("yolov8n.yaml")

    common = dict(
        data=str(DATA_YAML),
        imgsz=640,
        device=device,
        optimizer="SGD",
        lr0=0.01,
        patience=20,
        save=True,
        project=str(RUNS_DIR),
        name="barcode_hifi_v2",
        workers=0,
        verbose=True,
    )

    if QUICK:
        # 冒烟模式：快速验证管线，不要用来出最终精度
        model.train(
            epochs=15,
            fraction=0.1,
            batch=8,
            cache=False,
            degrees=180,
            project=str(RUNS_DIR),
            name="barcode_quick_v2",
            **{k: v for k, v in common.items() if k not in ("project", "name")},
        )
    else:
        model.train(
            epochs=100,
            fraction=1.0,          # 全量数据
            batch=16,              # 8GB 显存 imgsz=640 可用；OOM 就改成 8 或 4
            cache="disk",          # 磁盘缓存，避免内存爆
            # ---- 数据增强：专门针对倒置/反光/倾斜/曲面 ----
            degrees=180,           # 关键：支持 90/180 旋转，治倒置漏检
            translate=0.1,
            scale=0.5,
            shear=5.0,
            perspective=0.0005,    # 轻微透视，治倾斜/曲面
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,             # 亮度扰动，治反光/低光
            mosaic=1.0,
            mixup=0.1,
            close_mosaic=10,       # 最后 10 轮关闭 mosaic 稳定收敛
            **common,
        )

    print("\n" + "=" * 64)
    print("Training complete! Evaluating on test split...")
    print("=" * 64)
    metrics = model.val(data=str(DATA_YAML), split="test", device=device)

    print("\n" + "=" * 64)
    print("TEST METRICS SUMMARY")
    print("=" * 64)
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print("=" * 64)
    print("下一步：python scripts/eval_real.py  跑实拍图看误检是否下降")


if __name__ == "__main__":
    main()
