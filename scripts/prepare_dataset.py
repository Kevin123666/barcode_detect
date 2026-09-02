"""
Dataset download + clean (filter QR Code, keep only barcode class)
Dataset: https://www.kaggle.com/datasets/kushagrapandya/barcode-detection
Original classes: 0=Barcode, 1=QR Code
After clean: only 0=barcode
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
    for item in sorted(raw_path.iterdir()):
        print(f"  {item.name}/")
    yaml_path = raw_path / "data.yaml"
    if yaml_path.exists():
        print(f"\nOriginal data.yaml:")
        print(yaml_path.read_text(encoding="utf-8"))
    for split in ["train", "test", "valid"]:
        split_dir = raw_path / split
        if not split_dir.exists():
            continue
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        n_imgs = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbls = len(list(lbl_dir.glob("*"))) if lbl_dir.exists() else 0
        print(f"  {split:6s}: images={n_imgs}, labels={n_lbls}")
        qr_count, barcode_count = 0, 0
        if lbl_dir.exists():
            for lbl_file in lbl_dir.glob("*.txt"):
                for line in lbl_file.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    if cls_id == 0:
                        barcode_count += 1
                    elif cls_id == 1:
                        qr_count += 1
        print(f"    -> barcode labels={barcode_count}, QR labels={qr_count}")


def clean_and_copy(raw_path: Path, clean_dir: Path):
    print("\n" + "=" * 60)
    print("[3/3] Cleaning dataset (remove QR Code, keep only barcode)...")
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

        kept_images, total_qr_removed = 0, 0
        for img_file in img_dir.glob("*"):
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                continue
            lines = lbl_file.read_text().strip().splitlines()
            kept_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id == 0:
                    kept_lines.append(line)
                elif cls_id == 1:
                    total_qr_removed += 1
            if kept_lines:
                (clean_lbl_dir / lbl_file.name).write_text("\n".join(kept_lines) + "\n")
                shutil.copy2(img_file, clean_img_dir / img_file.name)
                kept_images += 1
        print(f"  {split:6s}: kept {kept_images} images, removed {total_qr_removed} QR labels")

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
