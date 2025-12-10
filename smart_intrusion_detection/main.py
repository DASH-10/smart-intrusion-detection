from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .behavior import BehaviorAnalyzer
from .config import VIDEO_SOURCE, load_config
from .detection import Detector
from .open_vocab import OpenVocabDetector
from .tracking import CentroidTracker
from .utils import FpsCounter, draw_fps, draw_zones, ensure_dirs, setup_logging
from .zones import PolygonZone, Zone, ZoneManager

try:
    import cv2
except ImportError:
    cv2 = None

polygon_zones: list[PolygonZone] = []
current_points: list[tuple[int, int]] = []
DISPLAY_SCALE: float = 1.8  # visualization scale factor


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
        print(f"[INFO] Using webcam index: {source}")
        return cap
    print(f"[ERROR] Could not open webcam index: {source}")
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(Path(__file__).resolve().parent.parent)
    if args.disable_open_vocab:
        cfg.model.model_type = "yolo"
    # No static restricted zone; use dynamic zones from detections.
    restricted_zone = None
    door_window_manager = ZoneManager(cfg.door_window_zones) if cfg.door_window_zones else None
    if args.save_video:
        cfg.output.save_video = True
    if args.no_screenshots:
        cfg.output.save_screenshots = False
    logger = setup_logging(cfg.output.log_path)

    if cv2 is None:
        logger.error("OpenCV is not installed. Install with `pip install opencv-python` and retry.")
        return 1

    ensure_dirs(
        [
            cfg.output.video_path.parent,
            cfg.output.screenshot_dir,
        ]
    )

    # YOLO-only detector for speed and stability.
    base_detector = Detector(cfg.model)
    frame_index = 0
    zone_manager = ZoneManager(cfg.zones)
    tracker = CentroidTracker(cfg.tracker.max_age, cfg.tracker.max_distance)
    restricted_zone_ids = [z.id for z in zone_manager.zones if z.restricted]
    behavior = BehaviorAnalyzer(cfg.behavior, restricted_zone_ids)
    fps_counter = FpsCounter()

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

    logger.info("Starting loop. Press 'q' to exit.")
    category_colors = {
        "person": (0, 255, 0),       # green
        "living": (255, 0, 255),     # magenta
        "object": (255, 255, 0),     # cyan
    }

    # Haar cascades for face/eye detection (used on person detections).
    face_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_eye.xml"))

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            logger.info("Video stream ended or failed to grab frame.")
            break

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

        detections = base_detector.detect(frame)

        # Build dynamic zones for detected doors/windows (rectangles from detections).
        dynamic_dw_zones: list[Zone] = []
        # With YOLO-only, this will typically be empty unless your weights include these classes.

        for det in detections:
            cls_name = det.get("class_name", "")
            if cls_name == "person":
                category = "person"
            elif cls_name in {"dog", "cat"}:
                category = "living"
            else:
                category = "object"

            color = category_colors.get(category, (255, 255, 0))  # default to cyan
            det["category"] = category
            det["color"] = color

            x1, y1, x2, y2 = det["bbox"]
            if restricted_zone and category == "person":
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if restricted_zone.contains_point((cx, cy)):
                    category = "intrusion"
                    color = (0, 0, 255)  # red for intrusion
                    det["category"] = category
                    det["color"] = color

            if category == "person" and polygon_zones:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                for z in polygon_zones:
                    if z.contains_point(cx, cy):
                        category = "intrusion"
                        color = (0, 0, 255)
                        det["category"] = category
                        det["color"] = color
                        det["zone_name"] = z.name
                        break

            if door_window_manager and category == "person":
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                zones_to_check = list(door_window_manager.zones) if door_window_manager else []
                zones_to_check.extend(dynamic_dw_zones)
                # Mark as suspicious when in any detected or configured door/window zone.
                for zone in zones_to_check:
                    if zone.contains_bbox((x1, y1, x2, y2)):
                        category = "suspicious"
                        color = (0, 0, 255)  # red highlight for suspicious near entry points
                        det["category"] = category
                        det["color"] = color
                        break

            label = f"{category}: {cls_name} {det['score']:.2f}"
            text_y = max(15, y1 - 10)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Single label per detection (remove/avoid any other text overlays).
            cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        tracks = list(zone_manager.tag_tracks(tracker.update(detections)))
        events = behavior.process(tracks)
        for event in events:
            logger.info(
                "Event=%s zone=%s track=%s dwell=%.1fs",
                event.event,
                event.zone_id,
                event.track_id,
                event.dwell_time or 0.0,
            )
            if cfg.output.save_screenshots:
                screenshot = cfg.output.screenshot_dir / f"event_{event.zone_id}_track{event.track_id}_{int(time.time())}.jpg"
                cv2.imwrite(str(screenshot), frame)

        draw_fps(frame, fps_counter.fps)

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

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    logger.info("Shutdown complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
