from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
except Exception as exc:  # pragma: no cover - optional dependency
    IsolationForest = None
    _SKLEARN_ERROR: Optional[Exception] = exc
else:
    _SKLEARN_ERROR = None


@dataclass
class _HistoryEntry:
    frame_index: int
    x: float
    y: float
    in_zone: bool


class TrajectoryAnomalyDetector:
    """
    Lightweight anomaly scorer for person trajectories using IsolationForest.
    Maintains per-track history and a rolling "normal" feature buffer that is refit periodically.
    """

    def __init__(
        self,
        *,
        window_size_frames: int = 60,
        min_samples_before_scoring: int = 30,
        contamination: float = 0.05,
        score_threshold: float = 0.65,
        refit_every_n_frames: int = 300,
        max_dataset_size: int = 800,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.window_size_frames = window_size_frames
        self.min_samples_before_scoring = min_samples_before_scoring
        self.contamination = contamination
        self.score_threshold = score_threshold
        self.refit_every_n_frames = max(1, refit_every_n_frames)
        self.max_dataset_size = max_dataset_size

        self.enabled = IsolationForest is not None
        if not self.enabled:
            self.logger.warning(
                "scikit-learn not available (%s); ML anomaly detection disabled.", _SKLEARN_ERROR
            )

        self.histories: Dict[int, Deque[_HistoryEntry]] = defaultdict(lambda: deque(maxlen=self.window_size_frames))
        self._dataset: Deque[List[float]] = deque(maxlen=self.max_dataset_size)
        self._last_refit_frame: int = 0
        self._last_cleanup_frame: int = 0
        self.model: Optional[IsolationForest] = None

    def update(self, tracks: Iterable, frame_index: int, now: float) -> Dict[int, dict]:
        """
        Update model state with the latest tracks and produce an anomaly score per track.
        Returns mapping track_id -> {"score": float | None, "is_suspicious": bool, "features": List[float] | None}
        """
        results: Dict[int, dict] = {}
        if not self.enabled:
            for track in tracks:
                results[getattr(track, "track_id", -1)] = {"score": None, "is_suspicious": False, "features": None}
            return results

        active_ids = set()
        for track in tracks:
            track_id = getattr(track, "track_id", None)
            if track_id is None:
                continue

            # Focus on people-like tracks only.
            cls_name = (getattr(track, "class_name", None) or getattr(track, "label", "") or "").lower()
            if cls_name != "person":
                results[track_id] = {"score": None, "is_suspicious": False, "features": None}
                continue

            active_ids.add(track_id)
            cx, cy = self._center(track.bbox)
            in_zone = bool(getattr(track, "zone_ids", []))
            history = self.histories[track_id]
            history.append(_HistoryEntry(frame_index=frame_index, x=cx, y=cy, in_zone=in_zone))

            if len(history) < self.min_samples_before_scoring:
                results[track_id] = {"score": None, "is_suspicious": False, "features": None}
                continue

            features = self._compute_features(history)
            score = None
            is_suspicious = False

            if self.model is None:
                # Seed the dataset until we have enough points to fit.
                self._dataset.append(features)
                results[track_id] = {"score": None, "is_suspicious": False, "features": features}
                continue

            try:
                score = self._score(features)
                is_suspicious = score >= self.score_threshold
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.warning("Anomaly scoring failed: %s", exc)
                score = None
                is_suspicious = False

            if score is None or not is_suspicious:
                # Treat non-suspicious samples as recent normal observations for future refits.
                self._dataset.append(features)

            results[track_id] = {"score": score, "is_suspicious": is_suspicious, "features": features}

        self._maybe_refit(frame_index)
        self._purge_stale(frame_index, active_ids)
        return results

    def _center(self, bbox) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)

    def _compute_features(self, history: Deque[_HistoryEntry]) -> List[float]:
        coords = np.array([(h.x, h.y) for h in history], dtype=np.float32)
        frames = np.array([h.frame_index for h in history], dtype=np.float32)
        deltas = np.diff(coords, axis=0)
        frame_steps = np.diff(frames)
        frame_steps[frame_steps == 0] = 1.0

        step_distances = np.linalg.norm(deltas, axis=1)
        speeds = step_distances / frame_steps
        mean_speed = float(np.mean(speeds)) if speeds.size else 0.0
        max_speed = float(np.max(speeds)) if speeds.size else 0.0
        var_speed = float(np.var(speeds)) if speeds.size else 0.0

        total_displacement = float(np.linalg.norm(coords[-1] - coords[0])) if len(coords) >= 2 else 0.0
        path_length = float(np.sum(step_distances)) if step_distances.size else 0.0

        direction_changes = 0.0
        if len(deltas) >= 2:
            # Normalize by frame delta to approximate per-frame direction changes.
            vectors = deltas / frame_steps[:, None]
            for i in range(1, len(vectors)):
                v_prev, v_cur = vectors[i - 1], vectors[i]
                norm_prev = np.linalg.norm(v_prev)
                norm_cur = np.linalg.norm(v_cur)
                if norm_prev < 1e-3 or norm_cur < 1e-3:
                    continue
                cos_angle = float(np.clip(np.dot(v_prev, v_cur) / (norm_prev * norm_cur + 1e-8), -1.0, 1.0))
                angle = math.degrees(math.acos(cos_angle))
                if angle >= 45.0:
                    direction_changes += 1.0

        time_in_zone_ratio = float(sum(1 for h in history if h.in_zone)) / float(len(history))

        return [
            mean_speed,
            max_speed,
            var_speed,
            total_displacement,
            path_length,
            direction_changes,
            time_in_zone_ratio,
        ]

    def _score(self, features: List[float]) -> float:
        """Return suspicion score in [0, 1]; higher is more anomalous."""
        decision = float(self.model.decision_function([features])[0])
        raw = -decision  # invert so higher -> more anomalous
        # Logistic squash to [0, 1]
        return float(1.0 / (1.0 + math.exp(-raw)))

    def _maybe_refit(self, frame_index: int) -> None:
        if len(self._dataset) < self.min_samples_before_scoring:
            return
        if self.model is None or (frame_index - self._last_refit_frame) >= self.refit_every_n_frames:
            try:
                model = IsolationForest(
                    contamination=self.contamination,
                    n_estimators=100,
                    random_state=42,
                )
                model.fit(list(self._dataset))
                self.model = model
                self._last_refit_frame = frame_index
                self.logger.debug("IsolationForest refit on %d samples (frame %d).", len(self._dataset), frame_index)
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.warning("IsolationForest refit failed: %s", exc)

    def _purge_stale(self, frame_index: int, active_ids: set[int]) -> None:
        """Remove histories for tracks not seen in a while to bound memory."""
        if frame_index - self._last_cleanup_frame < self.window_size_frames:
            return
        stale_ids = [tid for tid, hist in self.histories.items() if tid not in active_ids and hist]
        for tid in stale_ids:
            self.histories.pop(tid, None)
        self._last_cleanup_frame = frame_index
