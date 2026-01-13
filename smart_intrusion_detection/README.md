# Smart Intrusion & Suspicious Behavior Detection

Real-time, desktop-only surveillance that combines pre-trained YOLO detections, OpenCV processing, and AI behavior logic to flag intrusions and suspicious dwell time. Runs locally with GPU acceleration (CUDA/FP16) when available; no browser or cloud pieces.

## Project Overview
- Real-time intelligent surveillance application that ingests webcam or video files.
- Deep learning-based detection (YOLO) plus optional open-vocabulary detection for doors/windows (GroundingDINO-ready).
- Image processing and visualization via OpenCV for drawing, tracking display, and UI interactions.
- AI behavior analysis for intrusion, dwell-time/loitering, and zone-based risk.
- Uses pre-trained models; the focus is on system design, integration, and pipeline orchestration rather than model training.

## System Architecture
```
Video Input
 -> Detection (YOLO / optional GroundingDINO)
 -> Tracking
 -> Zone Analysis (interactive polygons)
 -> Behavior Logic (intrusion + dwell time)
 -> Events & Logs (screenshots, JSONL, CSV, video)
```

## Key Features
- **Object detection (YOLOv8):** Pre-trained person/object detector; configurable thresholds and classes.
- **Open-vocabulary detection (GroundingDINO, optional):** Detect doors/windows/objects from text prompts; run every N frames for speed.
- **Tracking:** Lightweight centroid tracker out of the box; config is structured for swapping in DeepSORT/ByteTrack if desired.
- **Polygon danger zones (interactive):** Draw zones live with the mouse; centroids inside a zone are marked as intrusions.
- **Behavior analysis:** Intrusion and dwell-time/loitering events with cooldowns to reduce alert noise.
- **ML anomaly scoring (IsolationForest):** Optional unsupervised anomaly score on person tracks to flag suspicious pacing/paths without labels.
- **GPU acceleration (CUDA + FP16):** Uses CUDA when available; FP16 inference for higher FPS on GPUs.
- **Event logging (JSONL + CSV):** `EventLogger` helper writes structured events for downstream analytics; standard app logs also recorded.
- **Desktop usage (no web interface):** OpenCV window for interaction; everything runs locally/offline.

## GPU / CUDA Support
- Uses PyTorch CUDA builds; defaults to `cuda:0` when available and falls back to CPU automatically.
- FP16 enabled by default on GPU for faster inference; disable in `ModelConfig.use_fp16` if needed.
- To run on GPU, install a CUDA-enabled PyTorch wheel and ensure your NVIDIA drivers match. Example (CUDA 11.8):  
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- To force a specific GPU from the command line: `set CUDA_VISIBLE_DEVICES=0 && py -m smart_intrusion_detection.main --source 0`
- To force CPU: set `ModelConfig.device = "cpu"` in `config.py`.

## Open-vocabulary setup (GroundingDINO)
- Enable by setting `ModelConfig.model_type = "open_vocab"` in `config.py` (default prompts: person, door, window). Disable at runtime with `--disable-open-vocab`.
- Place weights/config at `models/groundingdino/GroundingDINO_SwinT_OGC.py` and `models/groundingdino/groundingdino_swint_ogc.pth`, or override the paths in `ModelConfig`.
- Control cadence with `open_vocab_every_n_frames` (default 10) or `--open-vocab-every N`; detections are cached between runs to save FPS.

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
# Example for RTX 30-series (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## How to Run
- Webcam (default GPU/CPU auto-select):  
  `py -m smart_intrusion_detection.main --source 0`
- Video file:  
  `py -m smart_intrusion_detection.main --source data/videos/example.mp4`
- GPU mode (explicit):  
  `set CUDA_VISIBLE_DEVICES=0 && py -m smart_intrusion_detection.main --source 0`
- Performance toggles:
  - `--device cuda:0|cpu` to override the device.
  - `--fp16` / `--no-fp16` to force half-precision on/off (CUDA only).
  - `--skip N` to run detection every N frames and reuse cached detections between.
  - `--display-scale 1.2` (for example) to shrink the UI for speed.
- Open-vocabulary toggles:
  - `--disable-open-vocab` to run YOLO-only.
  - `--open-vocab-every N` to throttle GroundingDINO calls (cache reused between runs).
- ML anomaly toggles:
  - `--disable-ml-anomaly` to turn off IsolationForest scoring.
  - `--anomaly-threshold 0.65` to tweak when a track is labeled suspicious (0-1 range).
  - `--anomaly-contamination 0.05` to set expected anomaly ratio; `--anomaly-refit-every 300` to control refits.
- Tracking toggles: `--tracker centroid|deepsort|bytetrack` (falls back to centroid if deps are missing).
- Output toggles: `--save-video` to write annotated MP4s; `--no-screenshots` to skip event captures.
- UI helper: `--ui-panel` launches a tiny Tkinter control panel for start/stop + toggles.
- Zone persistence: `--save-zones-path results/zones/polygon_zones.json` to pick where polygons are saved/loaded.

## Controls & Interaction
- Mouse: left-click to add polygon points; right-click to finalize the current polygon. Draw multiple zones as needed.
- Keyboard:
  - `Enter`: finalize the in-progress polygon.
  - `c`: remove the last finalized zone and clear the current polygon.
  - `q`: exit the application window.
  - `s`: save drawn polygons to `results/zones/polygon_zones.json` (default path, change with `--save-zones-path`).
  - `l`: load polygons from the same path and re-draw them.
  - Save/load zones: define static polygons in `config.py` (`ZoneConfig`) or serialize the drawn `polygon_zones` for reuse between runs (desktop-only, no cloud sync).

## Output Structure
```
results/
  output_videos/      # annotated videos when --save-video is used
  screenshots/        # event screenshots (intrusion/dwell hits)
  logs/               # application logs + structured events (app.log, events.jsonl, events.csv)
  zones/              # optional saved polygon definitions for reuse (polygon_zones.json)
data/
  videos/             # input clips
  images/             # sample images (used by run_image.py)
models/               # pre-trained weights (YOLO, GroundingDINO)
```

## AI & CV Techniques Used
- Machine Learning: pre-trained YOLOv8 for detection; optional GroundingDINO for open-vocabulary queries.
- Image Processing: OpenCV for capture, drawing, UI events, Haar cascades for face/eye overlays.
- AI Logic: rule-based behavior engine for intrusion/dwell events, cooldowns, and zone tagging; centroid tracking for ID stability.

## Academic & Practical Value
- Demonstrates end-to-end ML + CV integration with GPU-accelerated inference and FP16 optimization.
- Shows production-style architecture: detection -> tracking -> zone/behavior logic -> structured event outputs.
- Suitable for university evaluation and AI engineer portfolios where system design and real-time performance matter.
- Desktop-local workflow (no web UI or cloud services), highlighting privacy-preserving intelligent surveillance.
