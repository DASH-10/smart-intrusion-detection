from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from .config import ModelConfig

try:
    from ultralytics import YOLO
except ImportError:  # Ultralytics is optional; we fall back to a no-op detector.
    YOLO = None


class PersonDetector:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        requested_device = self.cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = self._select_device(requested_device)
        self.fp16_enabled = bool(self.cfg.use_fp16 and ("cuda" in str(self.device)) and torch.cuda.is_available())
        print(f"[PersonDetector] torch.cuda.is_available={torch.cuda.is_available()}")
        print(f"[PersonDetector] Selected device: {self.device}")
        print(f"[PersonDetector] FP16 enabled: {self.fp16_enabled}")
        try:
            torch.backends.cudnn.benchmark = bool(self.cfg.cudnn_benchmark)
        except Exception:
            # cudnn may be missing on CPU-only installs; ignore.
            pass
        self.model = self._load_model()
        if self.model is not None:
            try:
                self.model.to(self.device)
            except Exception as exc:
                self.logger.error("Failed to move model to %s: %s", self.device, exc)

    def _select_device(self, requested: str) -> str:
        """Resolve the best device based on config and availability."""
        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available; falling back to CPU.")
            return "cpu"
        return requested

    def _load_model(self) -> Optional[YOLO]:
        if YOLO is None:
            self.logger.warning("Ultralytics YOLO not installed; PersonDetector will return empty detections.")
            return None

        try:
            return YOLO(self.cfg.weights)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Failed to load YOLO model '%s': %s", self.cfg.weights, exc)
            return None

    def detect(self, frame) -> List[Dict]:
        """Run YOLOv8 and return detections for all classes."""
        if self.model is None:
            return []

        use_half = bool(self.fp16_enabled)
        results = self.model(
            frame,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.device,
            half=use_half,
            verbose=False,
        )[0]

        detections: List[Dict] = []
        names = getattr(results, "names", {}) if results is not None else {}
        boxes = getattr(results, "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if self.cfg.allowed_classes and cls_id not in self.cfg.allowed_classes:
                continue
            class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            detections.append(
                {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "score": conf,
                    "class_id": cls_id,
                    "class_name": class_name,
                }
            )

        return detections

    # Backwards-compatible alias.
    def predict(self, frame) -> List[Dict]:
        return self.detect(frame)


# Keep existing name for compatibility
Detector = PersonDetector
