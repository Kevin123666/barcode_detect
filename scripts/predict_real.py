"""
Real-world barcode detection test
- Reads images from real_test/ (phone photos of products/packages)
- Runs YOLOv8n inference and saves annotated results
"""
from pathlib import Path

import ultralytics.utils.downloads as _dl
_dl.safe_download = lambda *a, **k: None

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST_PT = PROJECT_ROOT / "runs" / "barcode_fast_v3" / "weights" / "best.pt"
INPUT_DIR = PROJECT_ROOT / "real_test"
OUT_DIR = PROJECT_ROOT / "runs" / "real_predict"


def main():
    if not BEST_PT.exists():
        print(f"ERROR: model not found: {BEST_PT}")
        return

    imgs = [p for p in INPUT_DIR.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")]
    if not imgs:
        print(f"No images in {INPUT_DIR}. Put phone photos there first.")
        return

    model = YOLO(str(BEST_PT))

    results = model.predict(
        source=str(INPUT_DIR),
        conf=0.25,
        iou=0.45,
        device=0,
        imgsz=640,          # higher resolution for real photos
        project=str(OUT_DIR),
        name="results",
        save=True,
        save_txt=True,
        save_conf=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("REAL-WORLD DETECTION SUMMARY")
    print("=" * 60)
    for r in results:
        name = Path(r.path).name
        n = len(r.boxes) if r.boxes is not None else 0
        if n > 0:
            confs = [f"{c:.2f}" for c in r.boxes.conf.tolist()]
            print(f"  [OK] {name}: {n} barcode(s), conf={', '.join(confs)}")
        else:
            print(f"  [MISS] {name}: no barcode detected")
    print("=" * 60)
    print(f"Annotated images saved to: {OUT_DIR / 'results'}")


if __name__ == "__main__":
    main()
