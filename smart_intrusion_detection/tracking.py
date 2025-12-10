from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


def _centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    label: str
    score: float
    zone_ids: List[str] = field(default_factory=list)


class CentroidTracker:
    """Minimal centroid-based tracker to keep IDs stable without external deps."""

    def __init__(self, max_age: int = 30, max_distance: float = 80.0):
        self.max_age = max_age
        self.max_distance = max_distance
        self._tracks: Dict[int, Dict] = {}
        self._next_id = 1
        self._frame_index = 0

    def update(self, detections: Sequence[dict]) -> List[Track]:
        self._frame_index += 1
        unmatched_tracks = set(self._tracks.keys())
        assignments: Dict[int, dict] = {}

        for det in detections:
            bbox = det["bbox"]
            det_centroid = _centroid(bbox)
            track_id = self._match(det_centroid, unmatched_tracks)
            if track_id is None:
                track_id = self._register(det)
            else:
                unmatched_tracks.discard(track_id)
                self._update_track(track_id, det)
            assignments[track_id] = det

        # Age out tracks that were not matched this frame.
        for track_id in list(unmatched_tracks):
            track_state = self._tracks.get(track_id)
            if track_state:
                track_state["age"] += 1
                if track_state["age"] > self.max_age:
                    del self._tracks[track_id]

        return [self._to_track(track_id, det) for track_id, det in assignments.items()]

    def _match(self, centroid: Tuple[float, float], candidates: set[int]) -> Optional[int]:
        best_id = None
        best_dist = self.max_distance
        for track_id in candidates:
            track_centroid = self._tracks[track_id]["centroid"]
            dist = math.dist(track_centroid, centroid)
            if dist < best_dist:
                best_dist = dist
                best_id = track_id
        return best_id

    def _register(self, detection: dict) -> int:
        track_id = self._next_id
        self._next_id += 1
        self._tracks[track_id] = {
            "bbox": detection["bbox"],
            "centroid": _centroid(detection["bbox"]),
            "label": detection.get("class_name", str(detection.get("class_id", ""))),
            "score": detection["score"],
            "age": 0,
            "last_seen": self._frame_index,
        }
        return track_id

    def _update_track(self, track_id: int, detection: dict) -> None:
        self._tracks[track_id].update(
            {
                "bbox": detection["bbox"],
                "centroid": _centroid(detection["bbox"]),
                "label": detection.get("class_name", str(detection.get("class_id", ""))),
                "score": detection["score"],
                "age": 0,
                "last_seen": self._frame_index,
            }
        )

    def _to_track(self, track_id: int, detection: dict) -> Track:
        state = self._tracks[track_id]
        return Track(
            track_id=track_id,
            bbox=state["bbox"],
            label=state["label"],
            score=state["score"],
        )
