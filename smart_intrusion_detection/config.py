from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]

# Default webcam index (change to 1 or 2 if needed)
VIDEO_SOURCE = 0  # default to webcam index 1 (change if needed)


@dataclass
class ModelConfig:
    model_type: str = "yolo"  # "yolo" or "open_vocab"
    weights: str = "yolov8n.pt"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    imgsz: int = 640
    device: Optional[str] = None
    allowed_classes: Optional[Sequence[int]] = None
    source: Union[str, int] = "data/videos/example.mp4"
    # Open-vocabulary options (used when model_type == "open_vocab").
    open_vocab_prompts: List[str] = field(default_factory=lambda: ["person", "door", "window"])
    open_vocab_box_threshold: float = 0.25
    open_vocab_text_threshold: float = 0.25
    # Provide defaults you can replace with real paths to GroundingDINO config/weights.
    open_vocab_config_path: Optional[PathLike] = "models/groundingdino/GroundingDINO_SwinT_OGC.py"
    open_vocab_checkpoint_path: Optional[PathLike] = "models/groundingdino/groundingdino_swint_ogc.pth"


@dataclass
class ZoneConfig:
    id: str
    name: str
    points: List[Tuple[int, int]]
    restricted: bool = True


@dataclass
class TrackerConfig:
    max_age: int = 30
    max_distance: float = 80


@dataclass
class BehaviorConfig:
    dwell_time_seconds: float = 5.0
    cooldown_seconds: float = 10.0


@dataclass
class OutputConfig:
    save_video: bool = False
    save_screenshots: bool = True
    video_path: Path = Path("results/output_videos/output.mp4")
    screenshot_dir: Path = Path("results/screenshots")
    log_path: Path = Path("results/logs/app.log")


@dataclass
class AppConfig:
    project_root: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    zones: List[ZoneConfig] = field(default_factory=list)
    door_window_zones: List[ZoneConfig] = field(default_factory=list)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _default_zones() -> List[ZoneConfig]:
    # No default restricted zones; all zones are built dynamically from detections.
    return []


def _default_door_window_zones() -> List[ZoneConfig]:
    # No static door/window zones by default; will be created dynamically from detections.
    return []


def load_config(project_root: Optional[PathLike] = None) -> AppConfig:
    """Build the default application configuration."""
    # Default project root is the repo root (one level above this package).
    base = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    cfg = AppConfig(project_root=base)
    cfg.zones = _default_zones()
    cfg.door_window_zones = _default_door_window_zones()

    # Normalize common paths relative to the project root.
    cfg.model.source = str(base / cfg.model.source) if isinstance(cfg.model.source, str) else cfg.model.source
    if cfg.model.open_vocab_config_path:
        cfg.model.open_vocab_config_path = str((base / cfg.model.open_vocab_config_path).resolve())
    if cfg.model.open_vocab_checkpoint_path:
        cfg.model.open_vocab_checkpoint_path = str((base / cfg.model.open_vocab_checkpoint_path).resolve())
    cfg.output.video_path = (base / cfg.output.video_path).resolve()
    cfg.output.screenshot_dir = (base / cfg.output.screenshot_dir).resolve()
    cfg.output.log_path = (base / cfg.output.log_path).resolve()
    return cfg
