"""
v3 vs v2 实拍对比评估
============================================================================
对同一批实拍图分别跑旧模型(v3)和新模型(v2)，输出逐张对比表 + 并排可视化。

用法:
    python scripts/compare_models.py
    REAL_DIR=... python scripts/compare_models.py     # 自定义图片目录

输出:
    real_test_50/compare/           并排对比图 (左=v3, 右=v2)
    real_test_50/compare/compare.csv
    real_test_50/compare/summary.txt
"""
import os
import csv
import time
from pathlib import Path

import cv2
import numpy as np

# --- 离线补丁：跳过 ultralytics 的联网请求 ---
import ultralytics.utils.downloads as _dl
_orig_sd = _dl.safe_download


def _patch(url="", file=None, *a, **kw):
    if file and Path(file).exists() and Path(file).stat().st_size > 1000:
        return str(file)
    return str(file) if file else None


_dl.safe_download = _patch

from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent
REAL_DIR = Path(os.environ.get("REAL_DIR", PROJECT / "real_test_50" / "input"))
OUT_DIR = Path(os.environ.get("OUT_DIR", PROJECT / "real_test_50" / "compare"))

V3 = Path(os.environ.get("W_OLD", PROJECT / "runs" / "barcode_fast_v3" / "weights" / "best.pt"))
V2 = Path(os.environ.get("W_NEW", PROJECT / "runs" / "barcode_hifi_v4" / "weights" / "best.pt"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONF = float(os.environ.get("CONF", "0.25"))
MAX_SIDE = int(os.environ.get("MAX_SIDE", "1400"))   # 并排图最长边，控制内存


def draw(img, boxes, confs, title, color):
    """在图上画框 + 标题"""
    H, W = img.shape[:2]
    thick = max(3, W // 200)
    fs = max(0.5, W / 1100)
    ft = max(2, W // 600)
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        tag = f"{c:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        cv2.rectangle(img, (x1, max(0, y1 - th - 12)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(img, tag, (x1 + 5, max(th + 4, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft + 1)
    # 顶部标题条
    bh = max(30, H // 22)
    cv2.rectangle(img, (0, 0), (W, bh), color, -1)
    cv2.putText(img, title, (10, int(bh * 0.75)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.7, W / 700), (255, 255, 255), max(2, ft + 1))
    return img


def infer(model, img_path):
    r = model.predict(source=str(img_path), conf=CONF, verbose=False)[0]
    boxes, confs = [], []
    if r.boxes is not None and len(r.boxes):
        for b in r.boxes:
            boxes.append(b.xyxy[0].cpu().numpy())
            confs.append(float(b.conf[0].cpu().item()))
    return boxes, confs


def main():
    if not V3.exists():
        print(f"缺少 v3 权重: {V3}")
        return
    if not V2.exists():
        print(f"缺少 v2 权重: {V2}")
        print("v2 还没训练完，等 runs/barcode_hifi_v2/weights/best.pt 生成后再跑")
        return

    files = sorted([f for f in REAL_DIR.iterdir() if f.suffix.lower() in IMG_EXTS])
    if not files:
        print(f"{REAL_DIR} 里没有图片")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print(f"v3 vs v2 实拍对比  图片数={len(files)}  conf={CONF}")
    print(f"  v3: {V3}")
    print(f"  v2: {V2}")
    print("=" * 74)

    m3, m2 = YOLO(str(V3)), YOLO(str(V2))

    rows = []
    t0 = time.time()
    for i, p in enumerate(files, 1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [{i}] 读取失败 {p.name}")
            continue
        b3, c3 = infer(m3, p)
        b2, c2 = infer(m2, p)

        # 先缩放再并排，避免大图 hstack 内存爆掉
        h, w = img.shape[:2]
        sc = min(1.0, MAX_SIDE / max(h, w))
        if sc < 1.0:
            img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
            b3 = [bb * sc for bb in b3]
            b2 = [bb * sc for bb in b2]

        left = draw(img.copy(), b3, c3, f"v3  n={len(c3)}", (60, 60, 200))
        right = draw(img.copy(), b2, c2, f"v2  n={len(c2)}", (40, 160, 40))
        sep = np.full((img.shape[0], 6, 3), 255, np.uint8)
        cv2.imwrite(str(OUT_DIR / p.name), np.hstack([left, sep, right]))

        rows.append([
            p.name, len(c3), round(float(np.mean(c3)), 4) if c3 else 0.0,
            len(c2), round(float(np.mean(c2)), 4) if c2 else 0.0,
            len(c2) - len(c3),
        ])
        print(f"  [{i}/{len(files)}] {p.name:42s} v3={len(c3)}框/{np.mean(c3) if c3 else 0:.2f}"
              f"  v2={len(c2)}框/{np.mean(c2) if c2 else 0:.2f}")

    # CSV
    with open(OUT_DIR / "compare.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["文件名", "v3框数", "v3平均置信", "v2框数", "v2平均置信", "框数变化"])
        w.writerows(rows)

    n = len(rows)
    d3 = sum(1 for r in rows if r[1] > 0)
    d2 = sum(1 for r in rows if r[3] > 0)
    m3avg = float(np.mean([r[2] for r in rows if r[2] > 0])) if d3 else 0
    m2avg = float(np.mean([r[4] for r in rows if r[4] > 0])) if d2 else 0
    extra3 = sum(max(0, r[1] - 1) for r in rows)   # 多余框(疑似误检)
    extra2 = sum(max(0, r[3] - 1) for r in rows)

    summary = (
        "=== v3 vs v2 实拍对比 ===\n"
        f"图片总数      : {n}\n"
        f"v3 检出       : {d3}/{n} ({d3/n*100:.1f}%)   平均置信 {m3avg:.3f}\n"
        f"v2 检出       : {d2}/{n} ({d2/n*100:.1f}%)   平均置信 {m2avg:.3f}\n"
        f"v3 多余框合计 : {extra3}\n"
        f"v2 多余框合计 : {extra2}\n"
        f"漏检(v3->v2)  : {sum(1 for r in rows if r[1]==0 and r[3]>0)} 张被修复, "
        f"{sum(1 for r in rows if r[1]>0 and r[3]==0)} 张新增漏检\n"
        f"耗时          : {time.time()-t0:.1f}s\n"
    )
    print("\n" + summary)
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(f"并排对比图: {OUT_DIR}")
    print(f"对比表    : {OUT_DIR / 'compare.csv'}")


if __name__ == "__main__":
    main()
