from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .anomaly import TrajectoryAnomalyDetector
from .behavior import BehaviorAnalyzer
from .config import VIDEO_SOURCE, load_config
from .control_panel import ControlPanel, PanelState
from .detection import Detector
from .event_logger import EventLogger
from .open_vocab import OpenVocabDetector
from .tracker_adapter import build_tracker
from .utils import FpsCounter, draw_fps, draw_zones, ensure_dirs, setup_logging
from .zones import PolygonZone, Zone, ZoneManager
from .zones_io import load_polygon_zones, save_polygon_zones

try:
    import cv2
except ImportError:
    cv2 = None

polygon_zones: list[PolygonZone] = []
current_points: list[tuple[int, int]] = []
DISPLAY_SCALE: float = 1.8  # visualization scale factor


def _bbox_iou(box_a, box_b) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    denom = float(area_a + area_b - inter_area + 1e-6)
    return inter_area / denom if denom > 0 else 0.0


def _same_class(name_a: str | None, name_b: str | None) -> bool:
    if not name_a or not name_b:
        return False
    a, b = name_a.lower(), name_b.lower()
    if a == b:
        return True
    dw = {"door", "window"}
    return a in dw and b in dw


def _merge_detections(yolo_dets, ov_dets):
    merged = list(yolo_dets)
    for ov in ov_dets:
        keep_new = True
        for idx, base in enumerate(list(merged)):
            if _same_class(base.get("class_name"), ov.get("class_name")) and _bbox_iou(base["bbox"], ov["bbox"]) > 0.6:
                keep_new = ov.get("score", 0.0) > base.get("score", 0.0)
                if keep_new:
                    merged[idx] = ov
                break
        if keep_new:
            merged.append(ov)
    return merged


def _finalize_polygon():
    if len(current_points) >= 3:
        name = f"Zone {len(polygon_zones) + 1}"
        polygon_zones.append(PolygonZone(list(current_points), name=name))
    current_points.clear()


def on_mouse(event, x, y, flags, param):
    # Map from display coordinates back to original frame coordinates.
    gx = int(x / DISPLAY_SCALE)
    gy = int(y / DISPLAY_SCALE)
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((gx, gy))
    elif event == cv2.EVENT_RBUTTONDOWN:
        _finalize_polygon()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Intrusion Detection")
    parser.add_argument("--source", type=str, help="Video path or camera index (e.g., 0)", default=None)
    parser.add_argument("--save-video", action="store_true", help="Save annotated output to results/output_videos")
    parser.add_argument("--no-screenshots", action="store_true", help="Disable screenshot capture on events")
    parser.add_argument(
        "--disable-open-vocab",
        action="store_true",
        help="Disable open-vocabulary detector (use YOLO only for speed).",
    )
    parser.add_argument("--weights", type=str, default=None, help="YOLO weights path (e.g., yolov8s.pt)")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO input resolution (e.g., 768)")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="Detection IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g., cuda:0 or cpu")
    parser.add_argument("--fp16", dest="fp16", action="store_true", help="Enable fp16 on CUDA for YOLO")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable fp16 even on CUDA")
    parser.set_defaults(fp16=None)
    parser.add_argument("--tracker", choices=["centroid", "deepsort", "bytetrack"], help="Tracker backend")
    parser.add_argument("--open-vocab-every", type=int, default=None, help="Run open-vocab every N frames (cache in between)")
    parser.add_argument("--display-scale", type=float, default=None, help="UI display scale (lower for speed)")
    parser.add_argument("--skip", type=int, default=None, help="Run detection every N frames; reuse detections between.")
    parser.add_argument("--save-zones-path", type=str, default=None, help="Path to save/load polygon zones JSON.")
    parser.add_argument("--ui-panel", action="store_true", help="Launch Tkinter control panel for toggles.")
    parser.add_argument("--disable-ml-anomaly", action="store_true", help="Disable ML-based anomaly scoring.")
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=None,
        help="Suspicion score threshold (higher -> stricter), e.g., 0.65",
    )
    parser.add_argument(
        "--anomaly-contamination",
        type=float,
        default=None,
        help="Estimated anomaly proportion for IsolationForest (e.g., 0.05).",
    )
    parser.add_argument(
        "--anomaly-refit-every",
        type=int,
        default=None,
        help="Refit the anomaly model every N frames (e.g., 300).",
    )
    return parser.parse_args(argv)


def _coerce_source(source: str | int | None):
    if source is None:
        return None
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def open_video_capture(source):
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        print(f"[INFO] Using video source: {source}")
        return cap
    print(f"[ERROR] Could not open video source: {source}")
    return None


def main(argv: list[str] | None = None) -> int:
    global DISPLAY_SCALE, polygon_zones, current_points
    args = _parse_args(argv)
    cfg = load_config(Path(__file__).resolve().parent.parent)
    if args.disable_open_vocab:
        cfg.model.model_type = "yolo"
    if args.weights:
        cfg.model.weights = args.weights
    if args.imgsz is not None:
        cfg.model.imgsz = int(args.imgsz)
    if args.conf is not None:
        cfg.model.conf_threshold = float(args.conf)
    if args.iou is not None:
        cfg.model.iou_threshold = float(args.iou)
    if args.device:
        cfg.model.device = args.device
    if args.fp16 is not None:
        cfg.model.use_fp16 = args.fp16
    if args.tracker:
        cfg.tracker.tracker_type = args.tracker
    if args.open_vocab_every is not None:
        cfg.model.open_vocab_every_n_frames = max(1, args.open_vocab_every)
    if args.display_scale is not None:
        DISPLAY_SCALE = args.display_scale
        cfg.display_scale = args.display_scale
    if args.skip is not None:
        cfg.skip_frames = max(0, args.skip)
    if args.disable_ml_anomaly:
        cfg.anomaly.enable_ml_anomaly = False
    if args.anomaly_threshold is not None:
        cfg.anomaly.anomaly_threshold = float(args.anomaly_threshold)
    if args.anomaly_contamination is not None:
        cfg.anomaly.contamination = float(args.anomaly_contamination)
    if args.anomaly_refit_every is not None:
        cfg.anomaly.refit_every_n_frames = max(1, int(args.anomaly_refit_every))
    if args.save_video:
        cfg.output.save_video = True
    if args.no_screenshots:
        cfg.output.save_screenshots = False
    DISPLAY_SCALE = cfg.display_scale
    zones_save_path = Path(args.save_zones_path) if args.save_zones_path else cfg.project_root / "results/zones/polygon_zones.json"

    logger = setup_logging(cfg.output.log_path)

    if cv2 is None:
        logger.error("OpenCV is not installed. Install with `pip install opencv-python` and retry.")
        return 1

    logger.info("[INFO] torch cuda available: %s", torch.cuda.is_available())

    ensure_dirs(
        [
            cfg.output.video_path.parent,
            cfg.output.screenshot_dir,
            zones_save_path.parent,
            cfg.output.log_path.parent,
            cfg.project_root / "results/logs",
        ]
    )

    event_logger = EventLogger(cfg.project_root / "results/logs/events.jsonl", cfg.project_root / "results/logs/events.csv")

    base_detector = Detector(cfg.model)
    logger.info("[INFO] detector device: %s", getattr(base_detector, "device", cfg.model.device))

    ov_detector = None
    if cfg.model.model_type == "open_vocab" and not args.disable_open_vocab:
        candidate = OpenVocabDetector(cfg.model)
        if candidate.model is not None:
            ov_detector = candidate
            logger.info("Open-vocabulary detector initialized.")
        else:
            logger.warning("Open-vocabulary detector unavailable; continuing with YOLO only.")
    else:
        logger.info("Open-vocabulary detector disabled.")

    # No static restricted zone; use dynamic zones from detections.
    restricted_zone = None
    door_window_manager = ZoneManager(cfg.door_window_zones) if cfg.door_window_zones else None

    frame_index = 0
    zone_manager = ZoneManager(cfg.zones)
    tracker = build_tracker(cfg.tracker)
    restricted_zone_ids = [z.id for z in zone_manager.zones if z.restricted]
    behavior = BehaviorAnalyzer(cfg.behavior, restricted_zone_ids)
    anomaly_detector = None
    if cfg.anomaly.enable_ml_anomaly:
        candidate = TrajectoryAnomalyDetector(
            window_size_frames=cfg.anomaly.window_size_frames,
            min_samples_before_scoring=cfg.anomaly.min_samples_before_scoring,
            contamination=cfg.anomaly.contamination,
            score_threshold=cfg.anomaly.anomaly_threshold,
            refit_every_n_frames=cfg.anomaly.refit_every_n_frames,
        )
        if candidate.enabled:
            anomaly_detector = candidate
            logger.info(
                "ML anomaly detector enabled (threshold=%.2f, contamination=%.3f).",
                cfg.anomaly.anomaly_threshold,
                cfg.anomaly.contamination,
            )
        else:
            logger.warning("ML anomaly detector unavailable; continuing without it.")
    else:
        logger.info("ML anomaly detector disabled via config/CLI.")
    fps_counter = FpsCounter()
    last_detections: list[dict] = []
    last_yolo_detections: list[dict] = []
    ov_cached: list[dict] = []
    suspicious_last_log: dict[int, float] = {}
    intrusion_last_log: dict[int, float] = {}
    ml_suspicious_last_log: dict[int, float] = {}
    zone_lookup = {z.id: z.name for z in zone_manager.zones}

    panel_state = None
    if args.ui_panel:
        try:
            panel_state = PanelState(
                enable_open_vocab=ov_detector is not None,
                save_video=cfg.output.save_video,
                save_screenshots=cfg.output.save_screenshots,
                tracker_type=cfg.tracker.tracker_type,
                source=str(args.source) if args.source is not None else "",
            )
            panel = ControlPanel(panel_state)
            panel.start()
            logger.info("Control panel started (toggle start/stop without exiting).")
        except Exception as exc:  # pragma: no cover - optional UI path
            logger.warning("Control panel unavailable (%s); continuing without it.", exc)

    # Determine webcam index: CLI overrides config.
    source_arg = _coerce_source(args.source) if args.source is not None else None
    video_source = source_arg if source_arg is not None else VIDEO_SOURCE
    cap = open_video_capture(video_source)
    if cap is None:
        raise RuntimeError("No webcam available.")

    writer = None
    if cfg.output.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(cfg.output.video_path), fourcc, 20.0, (frame_w, frame_h))

    cv2.namedWindow("Smart Intrusion Detection", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Smart Intrusion Detection", on_mouse)

    if zones_save_path.exists():
        try:
            polygon_zones = load_polygon_zones(zones_save_path)
            logger.info("Loaded %d polygon zone(s) from %s", len(polygon_zones), zones_save_path)
        except Exception as exc:
            logger.warning("Failed to load saved zones: %s", exc)

    show_calibration = False
    logger.info("Starting loop. Press 'q' to exit, 's' to save zones, 'l' to load zones, 'k' to toggle calibration overlay.")
    category_colors = {
        "person": (0, 255, 0),       # green
        "living": (255, 0, 255),     # magenta
        "object": (255, 255, 0),     # cyan
        "intrusion": (0, 0, 255),
        "suspicious": (0, 0, 255),
    }

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            logger.info("Video stream ended or failed to grab frame.")
            break

        if panel_state and not panel_state.running:
            time.sleep(0.05)
            continue
        if panel_state:
            cfg.output.save_screenshots = panel_state.save_screenshots

        fps_counter.tick()
        frame_index += 1
        if restricted_zone:
            draw_zones(frame, [restricted_zone])
        if door_window_manager:
            draw_zones(frame, door_window_manager.zones)

        # Draw finalized polygon zones
        for z in polygon_zones:
            z.draw(frame, color=(255, 0, 0))
        # Draw in-progress polygon
        if len(current_points) >= 2:
            pts = np.array(current_points, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=False, color=(255, 0, 255), thickness=1)
            for (px, py) in current_points:
                cv2.circle(frame, (px, py), 3, (255, 0, 255), -1)

        interval = max(1, cfg.skip_frames) if cfg.skip_frames else 1
        detect_this_frame = ((frame_index - 1) % interval) == 0
        yolo_dets = last_yolo_detections
        if detect_this_frame or not last_yolo_detections:
            yolo_dets = base_detector.detect(frame)
            last_yolo_detections = yolo_dets

        ov_dets: list[dict] = []
        if ov_detector and (panel_state is None or panel_state.enable_open_vocab):
            ov_dets = ov_detector.detect_cached(frame, frame_index, cfg.model.open_vocab_every_n_frames)
            ov_cached = ov_dets
        elif ov_detector:
            ov_dets = ov_cached

        merged = _merge_detections(yolo_dets, ov_dets)
        if detect_this_frame or not last_detections:
            detections = merged
            last_detections = merged
        else:
            detections = last_detections

        # Build dynamic zones for detected doors/windows (rectangles from detections).
        dynamic_dw_zones: list[Zone] = []
        for idx, det in enumerate(detections):
            cls_lower = str(det.get("class_name", "")).lower()
            if cls_lower in {"door", "window"}:
                x1, y1, x2, y2 = det["bbox"]
                zone_id = f"dw_{frame_index}_{idx}"
                dynamic_dw_zones.append(
                    Zone(
                        id=zone_id,
                        name=det.get("class_name", "door/window"),
                        points=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                        restricted=False,
                    )
                )

        for det in detections:
            cls_name = det.get("class_name", "")
            if det.get("category"):
                category = det["category"]
            elif cls_name == "person":
                category = "person"
            elif cls_name in {"dog", "cat"}:
                category = "living"
            else:
                category = "object"

            color = category_colors.get(category, (255, 255, 0))  # default to cyan
            det["category"] = category
            det["color"] = color

            x1, y1, x2, y2 = det["bbox"]
            feet_x = (x1 + x2) // 2
            feet_y = y2
            if restricted_zone and category == "person":
                if restricted_zone.contains_point((feet_x, feet_y)):
                    category = "intrusion"
                    color = (0, 0, 255)  # red for intrusion
                    det["category"] = category
                    det["color"] = color

            if category == "person" and polygon_zones:
                for z in polygon_zones:
                    if z.contains_point(feet_x, feet_y):
                        category = "intrusion"
                        color = (0, 0, 255)
                        det["category"] = category
                        det["color"] = color
                        det["zone_name"] = z.name
                        det["zone_id"] = z.name
                        break

            if door_window_manager and category == "person":
                zones_to_check = list(door_window_manager.zones) if door_window_manager else []
                zones_to_check.extend(dynamic_dw_zones)
                # Mark as suspicious when in any detected or configured door/window zone.
                for zone in zones_to_check:
                    if zone.contains_bbox((x1, y1, x2, y2)):
                        category = "suspicious"
                        color = (0, 0, 255)  # red highlight for suspicious near entry points
                        det["category"] = category
                        det["color"] = color
                        det["zone_id"] = zone.id
                        det["zone_name"] = zone.name
                        break

            label = f"{category}: {cls_name} {det['score']:.2f}"
            text_y = max(15, y1 - 10)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Single label per detection (remove/avoid any other text overlays).
            cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        tracks = list(zone_manager.tag_tracks(tracker.update(detections, frame)))
        tracks_by_id = {t.track_id: t for t in tracks}
        now_ts = time.time()
        anomaly_results: dict[int, dict] = {}
        if anomaly_detector:
            anomaly_results = anomaly_detector.update(tracks, frame_index, now_ts)

        events = behavior.process(tracks, now=now_ts)
        for event in events:
            logger.info(
                "Event=%s zone=%s track=%s dwell=%.1fs",
                event.event,
                event.zone_id,
                event.track_id,
                event.dwell_time or 0.0,
            )
            screenshot = None
            if cfg.output.save_screenshots:
                screenshot = cfg.output.screenshot_dir / f"event_{event.zone_id}_track{event.track_id}_{int(time.time())}.jpg"
                cv2.imwrite(str(screenshot), frame)
            track = tracks_by_id.get(event.track_id)
            event_logger.log_event(
                event_type=event.event,
                track_id=event.track_id,
                zone_id=event.zone_id,
                zone_name=zone_lookup.get(event.zone_id, event.zone_id),
                dwell_time=event.dwell_time,
                bbox=track.bbox if track else None,
                class_name=(track.class_name if track else None) or (track.label if track else None),
                score=track.score if track else None,
                anomaly_score=None,
                source=str(video_source),
                screenshot_path=screenshot,
                frame_index=frame_index,
                category=track.category if track else None,
            )

        for track in tracks:
            if track.category == "intrusion":
                last_ts = intrusion_last_log.get(track.track_id, 0.0)
                if now_ts - last_ts > 2.0:
                    intrusion_last_log[track.track_id] = now_ts
                    screenshot = None
                    if cfg.output.save_screenshots:
                        screenshot = cfg.output.screenshot_dir / f"intrusion_track{track.track_id}_{int(now_ts)}.jpg"
                        cv2.imwrite(str(screenshot), frame)
                    event_logger.log_event(
                        event_type="intrusion",
                        track_id=track.track_id,
                        zone_id=track.zone_id or (track.zone_ids[0] if getattr(track, "zone_ids", []) else None),
                        zone_name=track.zone_name,
                        dwell_time=None,
                        bbox=track.bbox,
                        class_name=track.class_name or track.label,
                        score=track.score,
                        anomaly_score=None,
                        source=str(video_source),
                        screenshot_path=screenshot,
                        frame_index=frame_index,
                        category=track.category,
                    )

        for track in tracks:
            if track.category == "suspicious":
                last_ts = suspicious_last_log.get(track.track_id, 0.0)
                if now_ts - last_ts > 2.0:
                    suspicious_last_log[track.track_id] = now_ts
                    screenshot = None
                    if cfg.output.save_screenshots:
                        screenshot = cfg.output.screenshot_dir / f"suspicious_track{track.track_id}_{int(now_ts)}.jpg"
                        cv2.imwrite(str(screenshot), frame)
                    event_logger.log_event(
                        event_type="suspicious",
                        track_id=track.track_id,
                        zone_id=track.zone_id,
                        zone_name=track.zone_name,
                        dwell_time=None,
                        bbox=track.bbox,
                        class_name=track.class_name or track.label,
                        score=track.score,
                        anomaly_score=None,
                        source=str(video_source),
                        screenshot_path=screenshot,
                        frame_index=frame_index,
                        category=track.category,
                    )

        for track in tracks:
            ml_result = anomaly_results.get(track.track_id) if anomaly_results else None
            if not ml_result or ml_result.get("score") is None:
                continue
            if not ml_result.get("is_suspicious"):
                continue

            x1, y1, x2, y2 = track.bbox
            label = f"SUSPICIOUS(ML) score={ml_result['score']:.2f}"
            cv2.putText(frame, label, (x1, max(15, y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            last_ts = ml_suspicious_last_log.get(track.track_id, 0.0)
            if now_ts - last_ts < 10.0:
                continue
            ml_suspicious_last_log[track.track_id] = now_ts

            screenshot = None
            if cfg.output.save_screenshots:
                screenshot = cfg.output.screenshot_dir / f"ml_suspicious_track{track.track_id}_{int(now_ts)}.jpg"
                cv2.imwrite(str(screenshot), frame)

            zone_id = track.zone_id or (track.zone_ids[0] if getattr(track, "zone_ids", []) else None)
            zone_name = track.zone_name or (zone_lookup.get(zone_id) if zone_id else None)
            event_logger.log_event(
                event_type="ml_suspicious",
                track_id=track.track_id,
                zone_id=zone_id,
                zone_name=zone_name,
                dwell_time=None,
                bbox=track.bbox,
                class_name=track.class_name or track.label,
                score=track.score,
                anomaly_score=ml_result.get("score"),
                source=str(video_source),
                screenshot_path=screenshot,
                frame_index=frame_index,
                category=track.category,
                extra={"anomaly_features": ml_result.get("features")},
            )

        person_count = sum(
            1
            for det in detections
            if str(det.get("class_name", "")).lower() == "person" or det.get("class_id") == 0
        )
        current_fps = fps_counter.fps

        if show_calibration:
            overlay_lines = [
                f"conf: {cfg.model.conf_threshold:.2f}",
                f"imgsz: {cfg.model.imgsz}",
                f"persons: {person_count}",
                f"FPS: {current_fps:.1f}",
            ]
            for idx, text in enumerate(overlay_lines):
                cv2.putText(
                    frame,
                    text,
                    (10, 25 + idx * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        else:
            draw_fps(frame, current_fps)

        if writer:
            writer.write(frame)

        display_frame = cv2.resize(
            frame,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_LINEAR,
        )
        cv2.imshow("Smart Intrusion Detection", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (13, 10):  # Enter
            _finalize_polygon()
        elif key == ord("c"):  # Remove last finalized zone
            if polygon_zones:
                polygon_zones.pop()
            current_points.clear()
        elif key == ord("s"):
            try:
                path = save_polygon_zones(polygon_zones, zones_save_path)
                logger.info("Saved %d polygon zone(s) to %s", len(polygon_zones), path)
            except Exception as exc:
                logger.error("Failed to save zones: %s", exc)
        elif key == ord("l"):
            try:
                polygon_zones = load_polygon_zones(zones_save_path)
                current_points.clear()
                logger.info("Loaded %d polygon zone(s) from %s", len(polygon_zones), zones_save_path)
            except Exception as exc:
                logger.error("Failed to load zones: %s", exc)
        elif key == ord("k"):
            show_calibration = not show_calibration
            logger.info("Calibration overlay %s", "ON" if show_calibration else "OFF")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    logger.info("Shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
