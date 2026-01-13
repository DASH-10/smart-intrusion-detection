from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import ensure_dirs


class EventLogger:
    """Write structured events to JSONL + CSV for downstream processing."""

    def __init__(self, jsonl_path: Path, csv_path: Path):
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path)
        ensure_dirs([self.jsonl_path.parent, self.csv_path.parent])
        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        if not self.csv_path.exists():
            header = [
                "timestamp",
                "event_type",
                "track_id",
                "zone_id",
                "zone_name",
                "dwell_time",
                "bbox",
                "class_name",
                "score",
                "anomaly_score",
                "source",
                "screenshot_path",
                "frame_index",
                "category",
            ]
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def log_event(
        self,
        *,
        event_type: str,
        track_id: Optional[int],
        zone_id: Optional[str],
        zone_name: Optional[str],
        dwell_time: Optional[float],
        bbox: Optional[tuple],
        class_name: Optional[str],
        score: Optional[float],
        anomaly_score: Optional[float] = None,
        source: Optional[str],
        screenshot_path: Optional[Path],
        frame_index: Optional[int],
        category: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        ts = datetime.utcnow().isoformat()
        bbox_list = list(bbox) if bbox is not None else None
        payload: Dict[str, Any] = {
            "timestamp": ts,
            "event_type": event_type,
            "track_id": track_id,
            "zone_id": zone_id,
            "zone_name": zone_name,
            "dwell_time": dwell_time,
            "bbox": bbox_list,
            "class_name": class_name,
            "score": score,
            "anomaly_score": anomaly_score,
            "source": source,
            "screenshot_path": str(screenshot_path) if screenshot_path else None,
            "frame_index": frame_index,
            "category": category,
        }
        if extra:
            payload.update(extra)

        with open(self.jsonl_path, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(payload) + "\n")

        row = [
            payload["timestamp"],
            payload["event_type"],
            payload["track_id"],
            payload["zone_id"],
            payload["zone_name"],
            payload["dwell_time"],
            payload["bbox"],
            payload["class_name"],
            payload["score"],
            payload["anomaly_score"],
            payload["source"],
            payload["screenshot_path"],
            payload["frame_index"],
            payload["category"],
        ]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(row)
