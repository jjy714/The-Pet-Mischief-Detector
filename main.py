"""
Pet Mischief Detector — unified entry point.

Two execution modes:

  Eval mode  — batch-process a folder of static images, save annotated
               outputs to disk.  Used for Task 4B report visualizations
               and the bonus generalization test.

  Video mode — open a webcam or video file and run continuous real-time
               inference displayed in a window.  Uses a two-thread
               architecture so YOLO runs every frame while Depth
               Anything V2 updates asynchronously in the background.

Usage:
  uv run main.py --mode eval  [--input PATH] [--output PATH]
  uv run main.py --mode video [--source 0]  [--output PATH]

Options:
  --mode     eval | video
  --weights  path to YOLO weights  (default: model/runs/train/weights/best.pt)
  --input    image folder for eval mode
             (default: data/dataset/test/images)
  --output   output folder for annotated images / video
             (default: outputs/visualizations)
  --source   webcam index or video file path for video mode  (default: 0)
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).parent

from collections import deque

from model.detector import (
    _normalize_depth,
    _run_depth_model,
    fill_depths,
    infer_depth,
    infer_yolo,
    load_depth_model,
    load_yolo,
)
from model.mischief import _classify, calculate_mischief
from schema.Data import MischiefResult
from model.visualize import draw_frame

DEFAULT_WEIGHTS = ROOT / "model" / "runs" / "train" / "weights" / "best.pt"
DEFAULT_INPUT   = ROOT / "data" / "dataset" / "test" / "images"
DEFAULT_OUTPUT  = ROOT / "outputs" / "visualizations"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Eval mode
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace, yolo, processor, depth_model, device: str) -> None:
    input_dir = Path(args.input)
    out_dir   = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = (
        sorted(input_dir.glob("*.jpg"))
        + sorted(input_dir.glob("*.jpeg"))
        + sorted(input_dir.glob("*.png"))
    )
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(image_paths)} images from {input_dir} ...")
    counts: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARNING: could not read {img_path.name}")
            continue

        h, w = frame.shape[:2]
        detections = infer_yolo(yolo, frame)
        depth_map  = infer_depth(processor, depth_model, frame, device)
        detections = fill_depths(detections, depth_map, h, w)
        result     = calculate_mischief(detections, w, h, source=img_path.name)

        counts[result.risk_level] += 1
        annotated = draw_frame(frame, result, depth_map=depth_map)
        cv2.imwrite(str(out_dir / img_path.name), annotated)
        print(f"  [{result.risk_level:>6}]  {img_path.name}  —  {result.warning_message}")

    total = sum(counts.values())
    print(f"\n=== Summary ===")
    for level, count in counts.items():
        pct = 100.0 * count / total if total else 0.0
        print(f"  {level:<8}: {count:>5}  ({pct:.1f} %)")
    print(f"Output saved to {out_dir}")


# ---------------------------------------------------------------------------
# Video mode — two-thread architecture
# ---------------------------------------------------------------------------

class _SharedState:
    """
    Thread-safe container shared between the display thread and the depth
    inference thread.

    The display thread writes the latest captured frame and reads the
    latest depth map.  The depth thread reads the latest frame and writes
    the latest depth map.
    """

    def __init__(self) -> None:
        self._lock          = threading.Lock()
        self._frame: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._depth_count   = 0  # incremented each time depth is updated
        self._d_min_ema: float | None = None
        self._d_max_ema: float | None = None

    # Frame (written by display thread, read by depth thread)
    def set_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    # Depth (written by depth thread, read by display thread)
    def set_depth(self, depth: np.ndarray) -> None:
        with self._lock:
            self._depth = depth.copy()
            self._depth_count += 1

    def get_depth_and_count(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            d = self._depth.copy() if self._depth is not None else None
            return d, self._depth_count

    def set_depth_ema(self, raw: np.ndarray, alpha: float = 0.1) -> None:
        """
        Update depth using EMA-stabilized normalization.

        Maintains exponential moving averages of per-frame min/max so that
        the closeness scale stays consistent across frames rather than
        renormalizing independently each time.  Alpha = 0.1 gives ~1 second
        time constant at 10 FPS depth updates.
        """
        d_min = float(raw.min())
        d_max = float(raw.max())
        with self._lock:
            if self._d_min_ema is None:
                self._d_min_ema, self._d_max_ema = d_min, d_max
            else:
                self._d_min_ema = alpha * d_min + (1.0 - alpha) * self._d_min_ema
                self._d_max_ema = alpha * d_max + (1.0 - alpha) * self._d_max_ema
            self._depth = _normalize_depth(raw, self._d_min_ema, self._d_max_ema)
            self._depth_count += 1


def _depth_worker(
    state: _SharedState,
    processor,
    depth_model,
    device: str,
    stop: threading.Event,
) -> None:
    """
    Background thread: continuously reads the latest frame from shared
    state, runs Depth Anything V2, and writes the result back.

    Runs as fast as the depth model allows (~7–12 FPS on GPU for ViT-S).
    """
    while not stop.is_set():
        frame = state.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        raw = _run_depth_model(processor, depth_model, frame, device)
        state.set_depth_ema(raw)


def run_video(args: argparse.Namespace, yolo, processor, depth_model, device: str) -> None:
    source: str | int = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass  # treat as a video file path

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {args.source}")
        return

    # Optional: save output as a video file
    out_writer: cv2.VideoWriter | None = None
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        src_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_path  = out_dir / "mischief_video.mp4"
        out_writer = cv2.VideoWriter(
            str(vid_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            src_fps,
            (frame_w, frame_h),
        )
        print(f"Writing output video to {vid_path}")

    state    = _SharedState()
    stop_evt = threading.Event()

    depth_t = threading.Thread(
        target=_depth_worker,
        args=(state, processor, depth_model, device, stop_evt),
        daemon=True,
    )
    depth_t.start()

    # FPS tracking
    frame_times: list[float]       = []
    depth_update_ts: list[float]   = []
    prev_depth_count               = -1
    fps_display                    = 0.0
    fps_depth                      = 0.0

    # Temporal hysteresis: require sustained risk before emitting HIGH/MEDIUM.
    # Window of 15 frames at ~15–20 FPS ≈ 0.75–1 second.
    risk_history: deque[float] = deque(maxlen=15)

    print("Live detection running. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

            t0 = time.perf_counter()
            state.set_frame(frame)

            h, w = frame.shape[:2]

            # YOLO runs on every frame (fast path)
            detections = infer_yolo(yolo, frame)

            # Depth map comes from the background thread (may be one frame stale)
            depth_map, depth_count = state.get_depth_and_count()

            if depth_map is None:
                # Depth thread hasn't produced its first result yet — gate
                # mischief scoring to avoid false HIGH alerts from zero-depth.
                depth_map = np.zeros((h, w), dtype=np.float32)
                detections = fill_depths(detections, depth_map, h, w)
                result = MischiefResult(
                    source="video_frame",
                    detections=detections,
                    pairs=[],
                    max_risk_score=0.0,
                    risk_level="LOW",
                    warning_message="Initializing depth sensor...",
                )
            else:
                detections = fill_depths(detections, depth_map, h, w)
                result     = calculate_mischief(detections, w, h, source="video_frame")
                # Apply temporal hysteresis: classify by the minimum score in the
                # rolling window so that transient spikes don't fire HIGH alerts.
                risk_history.append(result.max_risk_score)
                conservative_max = min(risk_history)
                top_pair = result.pairs[0] if result.pairs else None
                level, warning = _classify(conservative_max, top_pair)
                result = result.model_copy(update={
                    "risk_level": level,
                    "warning_message": warning,
                })

            # Track depth update rate
            if depth_count != prev_depth_count:
                depth_update_ts.append(time.perf_counter())
                prev_depth_count = depth_count
                if len(depth_update_ts) > 30:
                    depth_update_ts.pop(0)
            if len(depth_update_ts) >= 2:
                span = depth_update_ts[-1] - depth_update_ts[0]
                fps_depth = (len(depth_update_ts) - 1) / span if span > 0 else 0.0

            # Track display FPS
            frame_times.append(time.perf_counter() - t0)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg = sum(frame_times) / len(frame_times)
            fps_display = 1.0 / avg if avg > 0 else 0.0

            annotated = draw_frame(
                frame, result,
                depth_map=depth_map,
                fps_display=fps_display,
                fps_depth=fps_depth,
            )

            cv2.imshow("Pet Mischief Detector", annotated)
            if out_writer is not None:
                out_writer.write(annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stop_evt.set()
        depth_t.join(timeout=3.0)
        cap.release()
        if out_writer is not None:
            out_writer.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pet Mischief Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run main.py --mode eval\n"
            "  uv run main.py --mode eval --input my_photos/ --output outputs/bonus\n"
            "  uv run main.py --mode video --source 0\n"
            "  uv run main.py --mode video --source recording.mp4 --output outputs/video\n"
        ),
    )
    parser.add_argument(
        "--mode", required=True, choices=["eval", "video"],
        help="eval: batch image mode; video: real-time webcam/video mode",
    )
    parser.add_argument(
        "--weights", default=str(DEFAULT_WEIGHTS),
        help="Path to YOLO weights (default: model/runs/train/weights/best.pt)",
    )
    parser.add_argument(
        "--input", default=str(DEFAULT_INPUT),
        help="[eval] Folder of input images",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help="Output folder for annotated images or video",
    )
    parser.add_argument(
        "--source", default="0",
        help="[video] Webcam index (0, 1, ...) or path to a video file",
    )
    return parser.parse_args()


def main() -> None:
    args   = _parse_args()
    device = get_device()
    print(f"Device  : {device}")

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights}\n"
            "Run scripts/03_train.py first."
        )

    print("Loading YOLO ...")
    yolo = load_yolo(str(weights))

    print("Loading Depth Anything V2 ...")
    processor, depth_model = load_depth_model(device)

    if args.mode == "eval":
        run_eval(args, yolo, processor, depth_model, device)
    else:
        run_video(args, yolo, processor, depth_model, device)


if __name__ == "__main__":
    main()
