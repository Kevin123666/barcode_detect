"""对比两个数据集的负样本（空标签）数量 —— 判断去重是否伤到了降误检能力"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def stat(name):
    d = ROOT / "data" / name / "train"
    imgs, lbls = d / "images", d / "labels"
    if not imgs.exists():
        return None
    pos = neg = 0
    for f in imgs.iterdir():
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        lp = lbls / (f.stem + ".txt")
        if not lp.exists():
            neg += 1
            continue
        txt = lp.read_text(encoding="utf-8", errors="ignore").strip()
        if len(txt) == 0:
            neg += 1
        else:
            pos += 1
    return pos, neg


for name in ("barcode_only", "barcode_dedup"):
    r = stat(name)
    if r:
        print(f"{name:16s} train 正样本 {r[0]:6d}   负样本 {r[1]:6d}")
