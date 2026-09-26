"""
断点续训 —— 从 last.pt 继续之前的训练
============================================================================
用户 2026-09-25 23:52 因断电手动停止了训练（当时进度 77/100 轮）。
本脚本从 last.pt 接着训，不用从头开始。

用法：
    python scripts/resume_train.py                                  # 默认续 runs/barcode_hifi_v4
    python scripts/resume_train.py runs/xxx/weights/last.pt         # 指定检查点

说明：
    ultralytics 的 resume=True 会自动读取该 run 目录下的 args.yaml，
    沿用原来的超参（imgsz=640 / degrees=180 / workers=4 等），
    并从上次中断的 epoch 继续，已完成的轮次不会重跑。
"""
import sys
from pathlib import Path

# --- 离线补丁：跳过 ultralytics 的联网请求 ---
import ultralytics.utils.downloads as _dl
import ultralytics.nn.tasks as _tasks

_orig_safe_download = _dl.safe_download


def _patched_safe_download(url="", file=None, *args, **kwargs):
    if file:
        p = Path(file)
        if p.exists() and p.stat().st_size > 100000:
            return str(p)
        try:
            return _orig_safe_download(url=url, file=file, *args, **kwargs)
        except Exception:
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_bytes(b"")
            return str(file)
    return None


_orig_torch_load = _tasks.torch_safe_load


def _patched_torch_load(file, device=None):
    try:
        return _orig_torch_load(file, device)
    except Exception:
        return None, None


_dl.safe_download = _patched_safe_download
_tasks.torch_safe_load = _patched_torch_load
# === 补丁结束 ===

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LAST = ROOT / "runs" / "barcode_hifi_v4" / "weights" / "last.pt"


def main():
    last = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LAST
    if not last.exists():
        print(f"找不到检查点: {last}")
        print("可用检查点：")
        for p in (ROOT / "runs").glob("*/weights/last.pt"):
            print(f"  {p}")
        return

    print("=" * 64)
    print(f"断点续训：{last}")
    print("=" * 64)
    model = YOLO(str(last))
    model.train(resume=True)

    print("\n续训完成，可在 runs/barcode_hifi_v4 查看 results.csv")


if __name__ == "__main__":
    main()
