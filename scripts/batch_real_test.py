"""
批量实拍测试 — 不分类，一个文件夹丢进去就行

用法: python scripts/batch_real_test.py
模型: runs/barcode_fast_v3/weights/best.pt
"""
import sys, csv, zipfile, time
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT / "real_test_50"
IN_DIR = TEST_DIR / "input"
OUT_DIR = TEST_DIR / "output"
WEIGHTS = PROJECT / "runs" / "barcode_fast_v3" / "weights" / "best.pt"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --- monkey-patch: 跳过 GitHub 下载 ---
import ultralytics.utils.downloads as _dl
def _patch(url="", file=None, *a, **kw):
    if file and Path(file).exists() and Path(file).stat().st_size > 1000:
        return str(file)
    return str(file) if file else None
_dl.safe_download = _patch
import ultralytics.utils.torch_utils as _tu
_tu.safe_download = _patch

from ultralytics import YOLO


def draw_boxes(img, boxes, confs):
    H, W = img.shape[:2]
    thick = max(4, W // 180)
    fs = max(0.6, W / 800)
    ft = max(2, W // 500)
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), thick)
        tag = f"barcode {c:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        cv2.rectangle(img, (x1, y1 - th - 18), (x1 + tw + 14, y1), (0, 200, 0), -1)
        cv2.putText(img, tag, (x1 + 7, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft + 1)
    return img


def main():
    if not WEIGHTS.exists():
        print(f"找不到模型权重: {WEIGHTS}")
        sys.exit(1)

    files = sorted([f for f in IN_DIR.iterdir() if f.suffix.lower() in IMG_EXTS])
    if not files:
        print("input 文件夹里没图片！把照片丢进 real_test_50/input/ 后再跑")
        sys.exit(1)

    print(f"加载模型: {WEIGHTS.name}")
    model = YOLO(str(WEIGHTS))
    print(f"共 {len(files)} 张图片\n")

    rows = []
    t0 = time.time()
    for i, img_p in enumerate(files, 1):
        img = cv2.imread(str(img_p))
        if img is None:
            print(f"  [{i}/{len(files)}] 读取失败: {img_p.name}")
            rows.append([img_p.name, "READ_FAIL", 0, 0.0, ""])
            continue

        r = model.predict(source=str(img_p), conf=0.25, verbose=False)[0]
        boxes, confs = [], []
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                boxes.append(xyxy)
                confs.append(float(b.conf[0].cpu().item()))

        vis = draw_boxes(img, boxes, confs)
        out_p = OUT_DIR / img_p.name
        cv2.imwrite(str(out_p), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

        n = len(confs)
        avg = float(np.mean(confs)) if confs else 0.0
        confs_str = ", ".join(f"{c:.2f}" for c in confs)
        status = "OK" if n >= 1 else "NO_DETECT"
        print(f"  [{i}/{len(files)}] {img_p.name:40s} {status:10s} boxes={n} conf={avg:.3f}")
        rows.append([img_p.name, status, n, round(avg, 4), confs_str])

    elapsed = time.time() - t0
    total = len(rows)
    ok = sum(1 for r in rows if r[1] == "OK")
    avg_conf_all = float(np.mean([r[3] for r in rows if r[3] > 0])) if ok else 0.0

    # CSV
    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["文件名", "状态", "检出框数", "平均置信度", "各框置信度"])
        w.writerows(rows)

    # 汇总
    summary = (
        f"=== 实拍测试汇总 ===\n"
        f"总图片数: {total}\n"
        f"检出数: {ok}/{total}  检出率 {ok/total*100:.1f}%\n"
        f"平均置信度: {avg_conf_all:.3f}\n"
        f"总耗时: {elapsed:.1f}s  平均 {elapsed/total:.1f}s/张\n"
    )
    print(f"\n{summary}")
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    # ZIP
    zip_path = TEST_DIR / f"real_test_result_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUT_DIR.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(OUT_DIR))
    print(f"打包完成: {zip_path}  ({zip_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
