"""
Dataset download + clean.
Dataset: https://www.kaggle.com/datasets/kushagrapandya/barcode-detection
Original classes: 0=Barcode, 1=QR Code

v2 改动（针对"误检多"）：
  - 原来把纯 QR / 无标签图全部删掉。现在把它们【保留为负样本】：
    图片照复制，标签文件留空（0 字节）。YOLO 训练时空标签 = 纯背景，
    让模型学会"QR 码 / 密集文字 / 纹理 都不是 1D barcode"，直接降误检。
  - 清洗后仍然只输出单类 0=barcode。
"""
import shutil
from pathlib import Path
import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "barcode_only"


def download_dataset():
    print("=" * 60)
    print("[1/3] Downloading kushagrapandya/barcode-detection...")
    print("=" * 60)
    path = kagglehub.dataset_download("kushagrapandya/barcode-detection")
    print(f"Downloaded to: {path}")
    return Path(path)


def inspect_dataset(raw_path: Path):
    print("\n" + "=" * 60)
    print("[2/3] Inspecting dataset structure...")
    print("=" * 60)
    yaml_path = raw_path / "data.yaml"
    if yaml_path.exists():
        print(yaml_path.read_text(encoding="utf-8"))
    for split in ["train", "test", "valid"]:
        split_dir = raw_path / split
        if not split_dir.exists():
            continue
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        n_imgs = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        print(f"  {split:6s}: images={n_imgs}")


def clean_and_copy(raw_path: Path, clean_dir: Path):
    print("\n" + "=" * 60)
    print("[3/3] Cleaning: barcode 保留为正样本，QR/无标签保留为【负样本】")
    print("=" * 60)
    clean_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test", "valid"]:
        raw_split = raw_path / split
        if not raw_split.exists():
            continue
        clean_img_dir = clean_dir / split / "images"
        clean_lbl_dir = clean_dir / split / "labels"
        clean_img_dir.mkdir(parents=True, exist_ok=True)
        clean_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_dir = raw_split / "images"
        lbl_dir = raw_split / "labels"

        pos, neg = 0, 0
        for img_file in img_dir.glob("*"):
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            kept_lines = []
            has_label = False
            if lbl_file.exists():
                for line in lbl_file.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    has_label = True
                    if int(parts[0]) == 0:           # 0=barcode 保留
                        kept_lines.append(line)
                    # 1=QR Code 不保留为正样本

            if kept_lines:
                # 正样本：有 barcode 框
                (clean_lbl_dir / lbl_file.name).write_text("\n".join(kept_lines) + "\n")
                pos += 1
            elif has_label:
                # 负样本 A：原来只有 QR 码 -> 图片复制，标签留空
                (clean_lbl_dir / lbl_file.name).write_text("")
                neg += 1
            else:
                # 负样本 B：无标签图 -> 图片复制，标签留空
                (clean_lbl_dir / lbl_file.name).write_text("")
                neg += 1
            shutil.copy2(img_file, clean_img_dir / img_file.name)

        print(f"  {split:6s}: 正样本 {pos} 张, 负样本 {neg} 张 (空标签背景)")

    yaml_content = f"""path: {clean_dir}
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  0: barcode
"""
    yaml_path = clean_dir / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\nNew data.yaml: {yaml_path}")
    print(yaml_content)
    return yaml_path


if __name__ == "__main__":
    raw = download_dataset()
    inspect_dataset(raw)
    yaml = clean_and_copy(raw, CLEAN_DIR)
    print(f"\nDone! data.yaml = {yaml}")
