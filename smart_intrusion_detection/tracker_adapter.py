from __future__ import annotations

import logging
from typing import List, Sequence

from .config import TrackerConfig
from .tracking import CentroidTracker, Track

logger = logging.getLogger(__name__)


class CentroidTrackerAdapter:
    """Thin wrapper that adapts CentroidTracker to the common interface."""

    def __init__(self, cfg: TrackerConfig):
        self.tracker = CentroidTracker(cfg.max_age, cfg.max_distance)

    def update(self, detections: Sequence[dict], frame=None) -> List[Track]:
        return self.tracker.update(detections)


class DeepSortAdapter:
    """Optional DeepSORT adapter (requires deep-sort-realtime)."""

    def __init__(self, cfg: TrackerConfig):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("deep-sort-realtime is not installed") from exc

        # Default params chosen for stability; override here if needed.
        self.tracker = DeepSort(max_age=cfg.max_age, max_iou_distance=0.7)

    def update(self, detections: Sequence[dict], frame=None) -> List[Track]:
        inputs = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_name = det.get("class_name", str(det.get("class_id", "")))
            inputs.append([x1, y1, x2, y2, det.get("score", 0.0), cls_name])

        tracks_out: List[Track] = []
        try:
            tracks = self.tracker.update_tracks(inputs, frame=frame)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("DeepSORT update failed (%s); falling back to centroid-like output.", exc)
            return []

        for t in tracks:
            if hasattr(t, "is_confirmed") and not t.is_confirmed():
                continue
            # deep-sort-realtime exposes helpers to_ltrb / to_tlbr; use whichever exists.
            bbox = None
            if hasattr(t, "to_ltrb"):
                bbox = t.to_ltrb()
            elif hasattr(t, "to_tlbr"):
                bbox = t.to_tlbr()
            elif hasattr(t, "tlbr"):
                bbox = t.tlbr
            if bbox is None:
                continue
            l, t0, r, b = bbox
            label = ""
            if hasattr(t, "get_det_class"):
                try:
                    label = str(t.get_det_class())
                except Exception:
                    label = ""
            score = 0.0
            if hasattr(t, "get_det_conf"):
                try:
                    score = float(t.get_det_conf())
                except Exception:
                    score = 0.0
            tracks_out.append(
                Track(
                    track_id=int(getattr(t, "track_id", -1)),
                    bbox=(int(l), int(t0), int(r), int(b)),
                    label=label or "obj",
                    score=score,
                    class_name=label or None,
                    category=None,
                )
            )
        return tracks_out


class ByteTrackAdapter:
    """Placeholder for ByteTrack; requires ultralytics tracking extras."""

    def __init__(self, cfg: TrackerConfig):
        # Avoid importing heavy dependencies unless explicitly requested.
        raise ImportError("ByteTrack optional dependency not installed; install tracking extras to enable.")

    def update(self, detections: Sequence[dict], frame=None) -> List[Track]:  # pragma: no cover - placeholder
        return []


def build_tracker(cfg: TrackerConfig):
    tracker_type = (cfg.tracker_type or "centroid").lower()
    if tracker_type == "deepsort":
        try:
            logger.info("Initializing DeepSORT tracker.")
            return DeepSortAdapter(cfg)
        except Exception as exc:
            logger.warning("DeepSORT unavailable (%s); falling back to centroid tracker.", exc)
    elif tracker_type == "bytetrack":
        try:
            logger.info("Initializing ByteTrack tracker.")
            return ByteTrackAdapter(cfg)
        except Exception as exc:
            logger.warning("ByteTrack unavailable (%s); falling back to centroid tracker.", exc)

    if tracker_type not in {"centroid", "deepsort", "bytetrack"}:
        logger.warning("Unknown tracker_type '%s'; defaulting to centroid.", tracker_type)
    return CentroidTrackerAdapter(cfg)

