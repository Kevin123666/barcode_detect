"""
Predict on test set using trained YOLOv8n model and save visualizations
"""
from pathlib import Path

# Lightweight patch only
import ultralytics.utils.downloads as _dl
_dl.safe_download = lambda *a, **k: None

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST_PT = PROJECT_ROOT / "runs" / "barcode_fast_v3" / "weights" / "best.pt"
TEST_DIR = PROJECT_ROOT / "data" / "barcode_only" / "test" / "images"
OUT_DIR = PROJECT_ROOT / "runs" / "predictions"


def main():
    if not BEST_PT.exists():
        print(f"ERROR: {BEST_PT} not found!")
        return

    model = YOLO(str(BEST_PT))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Predicting {TEST_DIR} ...")
    results = model.predict(
        source=str(TEST_DIR),
        conf=0.25,
        iou=0.45,
        device=0,
        project=str(OUT_DIR),
        name="barcode_test_results",
        save=True,
        save_txt=True,
        save_conf=True,
        verbose=True,
    )

    print(f"\nDone! Results saved to: {OUT_DIR / 'barcode_test_results'}")
    print(f"Total images: {len(results)}")

    total_dets = sum(len(r.boxes) for r in results if r.boxes is not None)
    print(f"Total barcode detections: {total_dets}")


if __name__ == "__main__":
    main()
