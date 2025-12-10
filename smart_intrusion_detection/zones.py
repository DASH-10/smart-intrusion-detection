from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .config import ZoneConfig

try:
    import cv2
except ImportError:
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise ImportError("OpenCV is required for zone geometry; install with `pip install opencv-python`.")


def _contour(points: List[Tuple[int, int]]) -> np.ndarray:
    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))


@dataclass
class Zone:
    id: str
    name: str
    points: List[Tuple[int, int]]
    restricted: bool = True

    def contains_point(self, point: Tuple[int, int]) -> bool:
        _require_cv2()
        contour = _contour(self.points)
        return cv2.pointPolygonTest(contour, point, False) >= 0

    def contains_bbox(self, bbox: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return self.contains_point((cx, cy))


class ZoneManager:
    def __init__(self, zone_configs: Sequence[ZoneConfig]):
        self.logger = logging.getLogger(__name__)
        self.zones: List[Zone] = [Zone(**zc.__dict__) for zc in zone_configs]
        self.logger.info("Loaded %d zone(s).", len(self.zones))

    def zones_for_bbox(self, bbox: Tuple[int, int, int, int]) -> List[Zone]:
        """Return zones that contain the bbox centroid."""
        return [zone for zone in self.zones if zone.contains_bbox(bbox)]

    def tag_tracks(self, tracks: Iterable) -> Iterable:
        """Attach zone_ids to track-like objects."""
        for track in tracks:
            zone_ids = [z.id for z in self.zones_for_bbox(track.bbox)]
            setattr(track, "zone_ids", zone_ids)
            yield track


class PolygonZone:
    def __init__(self, points, name="Zone"):
        """
        points: list of (x, y) tuples
        """
        _require_cv2()
        self.name = name
        self.points = np.array(points, dtype=np.int32)

    def contains_point(self, x, y) -> bool:
        result = cv2.pointPolygonTest(self.points, (float(x), float(y)), False)
        return result >= 0

    def draw(self, frame, color=(255, 0, 0)):
        cv2.polylines(frame, [self.points], isClosed=True, color=color, thickness=2)
        x0, y0 = self.points[0]
        cv2.putText(
            frame,
            self.name,
            (int(x0), int(y0) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
