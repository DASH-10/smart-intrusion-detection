from __future__ import annotations

import sys
from pathlib import Path

import cv2

# Make local imports work whether run as a script or module.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from config import ModelConfig
except ImportError:
    from smart_intrusion_detection.config import ModelConfig  # type: ignore

try:
    from detection import Detector as PersonDetector  # type: ignore
except ImportError:
    from smart_intrusion_detection.detection import Detector as PersonDetector  # type: ignore
try:
    from open_vocab import OpenVocabDetector  # type: ignore
except ImportError:
    try:
        from smart_intrusion_detection.open_vocab import OpenVocabDetector  # type: ignore
    except ImportError:
        OpenVocabDetector = None  # type: ignore

try:
    from utils import ensure_dirs
except ImportError:
    from smart_intrusion_detection.utils import ensure_dirs  # type: ignore


def _find_default_image(img_dir: Path) -> Path | None:
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        matches = list(img_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _draw_detections(frame, detections) -> None:
    category_colors = {
        "person": (0, 255, 0),       # green
        "living": (255, 0, 255),     # magenta
        "object": (255, 255, 0),     # cyan
    }
    for det in detections:
        cls_name = det.get("class_name", "")
        if cls_name == "person":
            category = "person"
        elif cls_name in {"dog", "cat"}:
            category = "living"
        else:
            category = "object"

        color = category_colors.get(category, (255, 255, 0))
        det["category"] = category
        det["color"] = color

        x1, y1, x2, y2 = det["bbox"]
        label = f"{category}: {cls_name} {det['score']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def main(image_path: str | None = None) -> int:
    project_root = Path(__file__).resolve().parent.parent

    model_cfg = ModelConfig()
    if isinstance(model_cfg.weights, str) and not Path(model_cfg.weights).is_absolute():
        model_cfg.weights = str(project_root / model_cfg.weights)

    img_dir = project_root / "data" / "images"
    img_path = Path(image_path) if image_path else _find_default_image(img_dir)
    if img_path is None or not img_path.exists():
        print(f"No image found. Provide a path or place an image under {img_dir}")
        return 1

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Failed to load image: {img_path}")
        return 1

    detections = []
    # Run open-vocab for doors/windows if available.
    if getattr(model_cfg, "model_type", "yolo") == "open_vocab" and OpenVocabDetector is not None:
        ov = OpenVocabDetector(model_cfg)
        if getattr(ov, "model", None) is not None:
            detections.extend(ov.detect(image))
    # Always run the base detector for people (and any other YOLO classes).
    base = PersonDetector(model_cfg)
    detections.extend(base.detect(image))
    _draw_detections(image, detections)

    output_path = project_root / "results" / "output_image.jpg"
    ensure_dirs([output_path.parent])
    cv2.imwrite(str(output_path), image)

    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(img_arg))
