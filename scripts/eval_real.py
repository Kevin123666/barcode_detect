"""
用训练好的模型跑【实拍难例】回归测试集，输出误检/漏检报告。
这就是老师说的"把容易出错的部分收集起来再训练"里的"再验证"环节。

用法：
    python scripts/eval_real.py
    可选：REAL_DIR=/path/to/images CONF=0.3 python scripts/eval_real.py
"""
import os
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHT_CANDIDATES = [
    PROJECT_ROOT / "runs" / "barcode_hifi_v2" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "barcode_fast_v3" / "weights" / "best.pt",
]
REAL_DIR = Path(os.environ.get("REAL_DIR", PROJECT_ROOT / "real_eval"))
CONF = float(os.environ.get("CONF", "0.30"))


def pick_weight() -> Path:
    for w in WEIGHT_CANDIDATES:
        if w.exists():
            return w
    raise FileNotFoundError(
        "找不到训练好的权重！请先 python scripts/train.py，"
        "或设置环境变量指向你的 best.pt"
    )


def main():
    weight = pick_weight()
    if not REAL_DIR.exists():
        raise FileNotFoundError(f"实拍目录不存在: {REAL_DIR}")

    imgs = sorted([p for p in REAL_DIR.glob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if not imgs:
        raise FileNotFoundError(f"{REAL_DIR} 里没有图片")

    print("=" * 64)
    print(f"实拍回归评估  权重={weight.name}  conf={CONF}")
    print(f"图片数={len(imgs)}  目录={REAL_DIR}")
    print("=" * 64)

    model = YOLO(str(weight))
    results = model.predict(
        source=str(REAL_DIR),
        conf=CONF,
        iou=0.45,
        device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu",
        verbose=False,
    )

    rows, total_conf, detected = [], 0.0, 0
    low_conf, multi_box = [], []
    for r in results:
        name = Path(r.path).name
        boxes = r.boxes
        n = len(boxes) if boxes is not None else 0
        confs = [float(c) for c in boxes.conf] if n else []
        avg = sum(confs) / n if n else 0.0
        total_conf += avg
        if n > 0:
            detected += 1
        rows.append((name, n, avg, confs))
        if 0 < n and avg < 0.5:
            low_conf.append(name)
        if n >= 3:
            multi_box.append((name, n))

    print(f"\n{'文件名':<48} {'框数':>4} {'均置信':>7}")
    print("-" * 64)
    for name, n, avg, _ in rows:
        flag = "  <- 低置信" if (0 < n and avg < 0.5) else \
               ("  <- 多框/误检?" if n >= 3 else ("  <- 漏检" if n == 0 else ""))
        print(f"{name:<48} {n:>4} {avg:>7.3f}{flag}")

    print("\n" + "=" * 64)
    print("汇总")
    print("=" * 64)
    print(f"  图片总数 : {len(imgs)}")
    print(f"  检出图片 : {detected}/{len(imgs)}  ({detected/len(imgs)*100:.1f}%)")
    print(f"  平均置信 : {total_conf/len(imgs):.3f}")
    print(f"  低置信检出(<0.5): {len(low_conf)} 张 -> {low_conf}")
    print(f"  多框可疑(>=3框) : {len(multi_box)} 张 -> {[m[0] for m in multi_box]}")
    print("=" * 64)
    print("提示：低置信/多框的图就是下一轮要收集进训练集的难例。")


if __name__ == "__main__":
    main()
