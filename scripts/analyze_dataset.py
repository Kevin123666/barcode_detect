"""
数据集重复度分析 —— 评估能否通过去重显著加速训练
============================================================================
Roboflow 导出时会对同一张原图做多份增强（文件名形如 base.rf.<hash>.jpg），
如果这些增强与 train.py 自带的 degrees=180 / HSV 增强高度重叠，
就可以每组只保留 1 张，数据量降数倍而几乎不损失多样性。

用法: python scripts/analyze_dataset.py
"""
import re
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "barcode_only"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RF = re.compile(r"\.rf\.[0-9a-f]{16,}$", re.I)


def base_of(stem: str) -> str:
    """去掉 Roboflow 的 .rf.<hash> 后缀，得到原图名"""
    return RF.sub("", stem, count=1)


def main():
    for split in ("train", "valid"):
        img_dir = DATA / split / "images"
        if not img_dir.exists():
            continue
        files = [f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
        groups = defaultdict(list)
        for f in files:
            groups[base_of(f.stem)].append(f)

        sizes = sorted((len(v) for v in groups.values()), reverse=True)
        total = len(files)
        uniq = len(groups)
        print("=" * 70)
        print(f"[{split}] 图片文件 {total} 张  ->  去重后 {uniq} 组  "
              f"(可减少 {100*(1-uniq/total):.1f}%)")
        print(f"  组大小分布: 最大 {sizes[0]}, 中位 {sizes[len(sizes)//2]}, "
              f"单张组 {sum(1 for s in sizes if s == 1)} 个")
        from collections import Counter
        for k, v in sorted(Counter(sizes).items())[:12]:
            print(f"    {k} 张/组 : {v} 组")
        print(f"  举例: 最大的 3 组")
        for name, fs in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:3]:
            print(f"    {name[:60]}  ({len(fs)} 张)")

        # 抽一组验证：同组内图片是否真的互为增强（内容不同但主体一致）
        multi = [v for v in groups.values() if len(v) >= 2]
        if multi:
            g = random.Random(0).choice(multi)
            g = sorted(g)[:4]
            print(f"  抽样组内 4 张的像素差异（越小越像，0 表示完全相同）:")
            ref = cv2.imread(str(g[0]))
            ref = cv2.resize(ref, (256, 256))
            for f in g[1:]:
                im = cv2.resize(cv2.imread(str(f)), (256, 256))
                diff = float(np.mean(np.abs(ref.astype(np.int16) - im.astype(np.int16))))
                print(f"    {f.name[:56]:58s} 平均像素差 {diff:6.1f} / 255")


if __name__ == "__main__":
    main()
