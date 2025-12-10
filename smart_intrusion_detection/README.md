# Smart Intrusion Detection (starter)

Lightweight scaffold for an intrusion detection pipeline built around YOLO + OpenCV, with pluggable tracking and zone/behavior logic.

## Quick start
- Create a virtualenv (recommended) and install deps:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  # For RTX 3060 (CUDA 11.8 wheels):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  pip install -r requirements.txt
  ```
- Put a test video in `data/videos/` or use a webcam.
- Run:
  ```bash
  py -m smart_intrusion_detection.main --source data/videos/example.mp4
  # or webcam
  py -m smart_intrusion_detection.main --source 0
  ```
  - Press `q` to exit.
  - Add `--save-video` to write annotated output to `results/output_videos/`.
  - Add `--no-screenshots` to skip event screenshots.

## Interactive polygon danger zones
- After the window opens, left-click to add points for a polygon.
- Right-click or press Enter to finalize the current polygon (zone). Repeat to create multiple zones.
- Zones are drawn (blue). The in-progress polygon shows in magenta.
- If a person's bbox center enters any polygon zone, it is marked as `intrusion` and drawn in red.

## File map
- `config.py` — defaults for model, zones, tracker, behavior, output.
- `detection.py` — YOLO wrapper; returns structured detections.
- `tracking.py` — minimal centroid tracker to keep IDs stable.
- `zones.py` — polygon zones + centroid-in-zone checks.
- `behavior.py` — intrusion/loitering detection with cooldowns.
- `utils.py` — logging, drawing helpers, FPS counter.
- `main.py` — wiring: loads config, runs loop, draws UI, saves outputs.

## Notes
- If Ultralytics is not installed, detection returns empty results so the loop still runs. Install it for real detections.
- Adjust zones in `config.py` (`_default_zones`) or pass `--source` to point at a different input.
