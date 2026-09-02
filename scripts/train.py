"""
Train YOLOv8n for barcode detection - OPTIMIZED FAST MODE
- imgsz=416, batch=16, amp=True (stable for 8GB VRAM)
- fraction=0.2 (5k images is enough for single class)
- cache='ram' (eliminate disk I/O bottleneck)
- epochs=20
"""
from pathlib import Path

# === Monkey-patches to avoid GitHub downloads ===
import ultralytics.utils.downloads as _dl
import ultralytics.nn.tasks as _tasks
import ultralytics.utils.torch_utils as _tu

def _patched_safe_download(url="", file=None, *args, **kwargs):
    if file:
        p = Path(file)
        if p.exists() and p.stat().st_size > 1000:
            return str(p)
        # Create empty file so downstream doesn't crash
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return str(file) if file else None

_orig_load = _tasks.torch_safe_load
def _patched_load(file, device=None):
    p = Path(file)
    if not p.exists() or p.stat().st_size < 100:
        return None, None
    try:
        return _orig_load(file, device)
    except Exception:
        return None, None

_dl.safe_download = _patched_safe_download
_tu.safe_download = _patched_safe_download
_tasks.torch_safe_load = _patched_load
# === End patches ===

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data" / "barcode_only" / "data.yaml"
RUNS_DIR = PROJECT_ROOT / "runs"


def main():
    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} not found! Run prepare_dataset.py first.")
        return

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("YOLOv8n Barcode Detection - FAST MODE v3")
    print("=" * 60)

    model = YOLO("yolov8n.yaml")

    results = model.train(
        data=str(DATA_YAML),
        epochs=20,
        imgsz=416,
        batch=16,             # Stable for 8GB VRAM
        device=0,
        amp=True,
        optimizer="SGD",
        lr0=0.01,
        workers=0,            # Windows multiprocessing has pickle issues
        fraction=0.2,         # Use only 20% of data (~5k images)
        cache="disk",         # Cache on disk (no pickle issues)
        project=str(RUNS_DIR),
        name="barcode_fast_v3",
        patience=8,
        save=True,
        save_period=10,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Training complete! Evaluating on test set...")
    print("=" * 60)

    metrics = model.val(data=str(DATA_YAML), split="test", device=0)

    print("\n" + "=" * 60)
    print("TEST METRICS SUMMARY")
    print("=" * 60)
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
