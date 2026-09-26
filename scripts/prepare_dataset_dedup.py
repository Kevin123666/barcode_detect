"""
构建去重数据集（加速训练）
============================================================================
Roboflow 导出时对同一张原图生成了多份离线增强（文件名 base.rf.<hash>.jpg），
train 里冗余度高达 84.4%（28696 -> 4470）。这些固定增强与 train.py 自带的
在线增强（degrees=180 / scale / translate / perspective / HSV / mosaic / mixup）
高度重叠，去掉可大幅缩短每轮耗时而几乎不损失信息量。

本脚本按「原图名」分组，每组只保留 1 张，用**硬链接**写入新目录
（同一分区，不额外占磁盘空间；失败则退回复制）。

用法: python scripts/prepare_dataset_dedup.py
输出: data/barcode_dedup/{train,valid,test}/{images,labels} + data.yaml
"""
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "barcode_only"
DST = ROOT / "data" / "barcode_dedup"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RF = re.compile(r"\.rf\.[0-9a-f]{16,}$", re.I)


def base_of(stem: str) -> str:
    return RF.sub("", stem, count=1)


def place(src: Path, dst: Path):
    """优先硬链接（零成本），失败则复制"""
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main():
    if not SRC.exists():
        print(f"源数据集不存在: {SRC}")
        return

    total_in = total_out = 0
    stat = {}
    for split in ("train", "valid", "test"):
        s_img, s_lbl = SRC / split / "images", SRC / split / "labels"
        if not s_img.exists():
            continue
        d_img, d_lbl = DST / split / "images", DST / split / "labels"
        d_img.mkdir(parents=True, exist_ok=True)
        d_lbl.mkdir(parents=True, exist_ok=True)

        groups = defaultdict(list)
        for f in s_img.iterdir():
            if f.suffix.lower() in IMG_EXTS:
                groups[base_of(f.stem)].append(f)

        n_in = sum(len(v) for v in groups.values())
        kept = 0
        n_missing_lbl = 0
        for base, fs in groups.items():
            rep = sorted(fs)[0]                       # 每组固定取第一张，可复现
            place(rep, d_img / rep.name)
            lbl = s_lbl / (rep.stem + ".txt")
            if lbl.exists():
                place(lbl, d_lbl / lbl.name)
            else:
                n_missing_lbl += 1                    # 负样本（无目标）属正常
            kept += 1

        total_in += n_in
        total_out += kept
        stat[split] = (n_in, kept, n_missing_lbl)
        print(f"[{split:5s}] {n_in:6d} -> {kept:6d} 张"
              f"  (减少 {100*(1-kept/max(n_in,1)):5.1f}%)"
              f"  无标签(负样本) {n_missing_lbl}")

    # 写 data.yaml
    yaml_text = (
        f"path: {DST}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: barcode\n"
    )
    (DST / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("-" * 60)
    print(f"合计 {total_in} -> {total_out} 张，减少 {100*(1-total_out/total_in):.1f}%")
    print(f"新数据集: {DST}")
    print(f"data.yaml: {DST / 'data.yaml'}")


if __name__ == "__main__":
    main()
