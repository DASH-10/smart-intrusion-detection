from __future__ import annotations

import logging
import random
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise ImportError("OpenCV is required for drawing; install with `pip install opencv-python`.")


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(log_path: Path) -> logging.Logger:
    ensure_dirs([log_path.parent])
    logger = logging.getLogger("smart_intrusion_detection")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def color_for_id(idx: int) -> Tuple[int, int, int]:
    random.seed(idx)
    return tuple(int(x) for x in (random.random() * 255, random.random() * 255, random.random() * 255))


def draw_zones(frame, zones: Sequence) -> None:
    _require_cv2()
    for zone in zones:
        contour = np.array(zone.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [contour], isClosed=True, color=(0, 165, 255), thickness=2)
        label_pos = (int(zone.points[0][0]), int(zone.points[0][1]) - 5)
        cv2.putText(frame, zone.name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)


def draw_tracks(frame, tracks: Sequence) -> None:
    _require_cv2()
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        color = color_for_id(track.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{track.label} #{track.track_id} {track.score:.2f}"
        if getattr(track, "zone_ids", []):
            label += f" | zones: {','.join(track.zone_ids)}"
        cv2.putText(frame, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def draw_fps(frame, fps: float) -> None:
    _require_cv2()
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


class FpsCounter:
    def __init__(self, window: int = 30):
        self.window = window
        self.samples: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self.samples.append(time.time())

    @property
    def fps(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        duration = self.samples[-1] - self.samples[0]
        if duration <= 0:
            return 0.0
        return (len(self.samples) - 1) / duration
