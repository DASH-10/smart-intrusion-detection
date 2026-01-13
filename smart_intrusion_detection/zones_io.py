from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .utils import ensure_dirs
from .zones import PolygonZone


def save_polygon_zones(polygon_zones: Iterable[PolygonZone], path: Path | str) -> Path:
    """Save polygon zones to JSON file."""
    target = Path(path)
    ensure_dirs([target.parent])
    payload = []
    for zone in polygon_zones:
        pts = [[int(x), int(y)] for x, y in zone.points.tolist()]
        payload.append({"name": zone.name, "points": pts})
    target.write_text(json.dumps(payload, indent=2))
    return target


def load_polygon_zones(path: Path | str) -> List[PolygonZone]:
    """Load polygon zones from JSON; returns empty list if file is missing."""
    source = Path(path)
    if not source.exists():
        return []
    data = json.loads(source.read_text())
    zones: List[PolygonZone] = []
    for idx, item in enumerate(data):
        name = item.get("name") or f"Zone {idx + 1}"
        points = item.get("points") or []
        zones.append(PolygonZone(points, name=name))
    return zones

