from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

from .config import ModelConfig

try:
    from groundingdino.util.inference import Model as GroundingDINOModel
except ImportError:
    GroundingDINOModel = None


class OpenVocabDetector:
    """GroundingDINO-based open-vocabulary detector (doors, windows, etc.)."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.model = self._load_model()

    def _load_model(self):
        if GroundingDINOModel is None:
            self.logger.warning("GroundingDINO is not installed; open-vocabulary detection will be disabled.")
            return None

        config_path = self.cfg.open_vocab_config_path or os.getenv("GROUNDINGDINO_CONFIG_PATH")
        checkpoint_path = self.cfg.open_vocab_checkpoint_path or os.getenv("GROUNDINGDINO_CHECKPOINT_PATH")
        if not config_path or not checkpoint_path:
            self.logger.error(
                "GroundingDINO config/checkpoint not provided. Set cfg.open_vocab_config_path / "
                "cfg.open_vocab_checkpoint_path or env vars GROUNDINGDINO_CONFIG_PATH / GROUNDINGDINO_CHECKPOINT_PATH."
            )
            return None

        config_path = str(Path(config_path).expanduser())
        checkpoint_path = str(Path(checkpoint_path).expanduser())

        try:
            return GroundingDINOModel(
                model_config_path=config_path,
                model_checkpoint_path=checkpoint_path,
                device=self.cfg.device or "cpu",
            )
        except Exception as exc:
            self.logger.error("Failed to load GroundingDINO: %s", exc)
            return None

    def detect(self, frame) -> List[Dict]:
        if self.model is None:
            return []

        try:
            preds = self.model.predict_with_classes(
                image=frame,
                classes=self.cfg.open_vocab_prompts,
                box_threshold=self.cfg.open_vocab_box_threshold,
                text_threshold=self.cfg.open_vocab_text_threshold,
            )
        except Exception as exc:
            self.logger.error("GroundingDINO inference failed: %s", exc)
            return []

        detections: List[Dict] = []

        # GroundingDINO may return a dict or a Supervision Detections object.
        if isinstance(preds, dict):
            boxes = preds.get("boxes", [])
            labels = preds.get("labels", [])
            scores = preds.get("scores", [])
        elif hasattr(preds, "xyxy"):
            boxes = getattr(preds, "xyxy", [])
            class_ids = getattr(preds, "class_id", [])
            scores = getattr(preds, "confidence", getattr(preds, "scores", []))
            labels = []
            for cid in class_ids:
                try:
                    labels.append(self.cfg.open_vocab_prompts[int(cid)])
                except Exception:
                    labels.append(str(cid))
        else:
            self.logger.error("Unexpected GroundingDINO output type: %s", type(preds))
            return []

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            detections.append(
                {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "score": float(score),
                    "class_id": -1,
                    "class_name": str(label),
                }
            )
        return detections

    # Backwards compatibility with Detector.predict
    def predict(self, frame) -> List[Dict]:
        return self.detect(frame)


if __name__ == "__main__":
    # Quick load/detect smoke test: `python -m smart_intrusion_detection.open_vocab`
    import numpy as np
    from .config import load_config

    cfg = load_config(Path(__file__).resolve().parent.parent)
    det = OpenVocabDetector(cfg.model)
    loaded = det.model is not None
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = det.detect(blank) if loaded else []
    print(f"open_vocab_loaded={loaded}")
    print(f"detections_on_blank={len(detections)}")
