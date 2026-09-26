"""统计 train 图片原始尺寸 —— 判断预缩放能否显著提速"""
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

try:
    from PIL import Image
except ImportError:
    print("需要 pillow")
    raise

for name in ("barcode_only", "barcode_dedup"):
    d = ROOT / "data" / name / "train" / "images"
    if not d.exists():
        continue
    files = [f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS]
    if not files:
        continue
    sample = random.Random(0).sample(files, min(600, len(files)))
    sizes = []
    for f in sample:
        try:
            with Image.open(f) as im:
                sizes.append(im.size)
        except Exception:
            pass
    maxside = [max(w, h) for w, h in sizes]
    print("=" * 66)
    print(f"[{name}] 抽样 {len(sizes)} / {len(files)} 张")
    print(f"  最长边: 中位 {sorted(maxside)[len(maxside)//2]}, "
          f"最大 {max(maxside)}, 最小 {min(maxside)}")
    for lo, hi in ((0, 481), (481, 641), (641, 1025), (1025, 2049), (2049, 99999)):
        n = sum(1 for m in maxside if lo <= m < hi)
        print(f"    {lo:>5}-{hi:<6} : {n:4d} 张 ({100*n/len(maxside):5.1f}%)")
    print(f"  最常见的 5 种尺寸: {Counter(sizes).most_common(5)}")
