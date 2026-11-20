"""Vehicle zone crossing demo for a fixed camera and pretrained COCO YOLO."""
import argparse
from contextlib import closing
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from getpass import getpass
import os
import time

import cv2
import numpy as np

from vehicle_event_log import EmailSettings, EventLog


ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "datasets/road-traffic-video-monitoring/british_highway_traffic.mp4"
# Normalized corners of a segment of the approaching carriageway.
DEFAULT_ZONE = [(0.61, 0.43), (0.87, 0.43), (0.78, 0.75), (0.41, 0.75)]
VEHICLE_CLASSES = [2, 3, 5, 7]
PARKING_VIDEO = ROOT / "datasets/ParkingVideo.mp4"
PARKING_ZONE = [(0.42, 0.28), (0.61, 0.28), (0.68, 0.58), (0.35, 0.58)]
WINDOW = "Vehicle monitor"
DATABASE = ROOT / "vehicle_events.sqlite3"
IMAGES = ROOT / "vehicle_images"


@dataclass(frozen=True)
class MonitorOptions:
    video: Path
    weights: str
    device: str | None
    initial: int | None
    read_plates: bool
    database: Path = DATABASE
    output: Path = IMAGES
    email: EmailSettings | None = field(default=None, repr=False)
    frame_limit: int | None = None


@dataclass(frozen=True)
class Preview:
    frame: np.ndarray
    entries: int
    exits: int
    inside: int
    visible: int
    video_seconds: float
    fps: float
    count_mismatch: bool


def preset_zone(frame, corners):
    height, width = frame.shape[:2]
    return np.round(np.array(corners) * (width, height)).astype(np.int32)



@dataclass
class Track:
    vehicle_id: int
    class_votes: Counter = field(default_factory=Counter)
    logged_class: str | None = None
    plate_reads: deque = field(default_factory=lambda: deque(maxlen=9))
    plate: str | None = None
    plate_sample_frame: int = 0
    inside: bool | None = None
    candidate_frames: int = 0
    candidate_side: bool | None = None
    last_frame: int = -2
    trail: deque = field(default_factory=lambda: deque(maxlen=8))

    def update(self, frame_number, point, distance, margin):
        if frame_number != self.last_frame + 1:
            self.candidate_frames = 0
            self.trail.clear()
        self.last_frame = frame_number
        self.trail.append(point)

        if abs(distance) <= margin:
            self.candidate_frames = 0
            return None
        inside = distance > 0
        if inside == self.inside:
            self.candidate_frames = 0
            return None
        if inside != self.candidate_side:
            self.candidate_frames = 0
            self.candidate_side = inside
        self.candidate_frames += 1
        if self.candidate_frames < 3:
            return None
        previous = self.inside
        self.inside = inside
        self.candidate_frames = 0
        if previous is None:
            return None
        return "entry" if inside else "exit"


def draw_zone(first):
    height, width = first.shape[:2]
    scale = min(1, 1050 / width, 650 / height)
    preview = cv2.resize(first, None, fx=scale, fy=scale)
    points = []

    def click(event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.imshow(WINDOW, preview)
    cv2.setMouseCallback(WINDOW, click)
    try:
        while True:
            canvas = preview.copy()
            if points:
                cv2.polylines(canvas, [np.array(points)], True, (0, 220, 255), 2)
            cv2.putText(canvas, "Click corners in order | Enter start | R reset | Esc cancel",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 220, 255), 2)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(25) & 0xff
            if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return None
            if key == ord("r"):
                points.clear()
            if key in (10, 13) and len(points) >= 3:
                polygon = np.round(np.array(points) / scale).astype(np.int32)
                if cv2.contourArea(polygon) >= 100:
                    return polygon
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=VIDEO)
    parser.add_argument("--weights", default="yolov8s.pt")
    parser.add_argument("--initial", type=int, help="Preset vehicles already inside (default: detect from first frame)")
    parser.add_argument("--draw-zone", action="store_true")
    parser.add_argument("--setup", action="store_true", help="Prompt for starting count and zone")
    parser.add_argument("--device", help="Use 0 for GPU or cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, help="Limit frames for short test runs")
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--output", type=Path, default=IMAGES)
    parser.add_argument("--parking", action="store_true", help="Run the parking video with plate recognition")
    args = parser.parse_args()
    if args.parking:
        args.video = PARKING_VIDEO
    if args.setup:
        answer = input("Starting vehicles [Enter = auto, or type a count]: ").strip()
        if answer:
            try:
                args.initial = int(answer)
            except ValueError:
                parser.error("Enter a whole number or leave the starting count blank.")
        args.draw_zone = input("Draw your own zone? [y/N]: ").strip().lower() == "y"
        if input("Enable Gmail alerts for this run? [y/N]: ").strip().lower() == "y":
            settings = {
                "VEHICLE_ALERT_SENDER": input("Sender Gmail address: ").strip(),
                "VEHICLE_ALERT_RECIPIENT": input("Recipient email address: ").strip(),
                "VEHICLE_ALERT_APP_PASSWORD": getpass("Gmail app password: ").replace(" ", ""),
            }
            if not all(settings.values()):
                parser.error("Enter the sender, recipient, and app password to enable email.")
            os.environ.update(settings)
        else:
            os.environ.pop("VEHICLE_ALERT_APP_PASSWORD", None)
    if (args.initial is not None and args.initial < 0) or (args.frames is not None and args.frames < 1):
        parser.error("Initial occupancy must be nonnegative and the frame limit positive.")
    if args.draw_zone and args.headless:
        parser.error("Drawing a zone requires the preview window.")
    if args.video.resolve() not in (VIDEO.resolve(), PARKING_VIDEO.resolve()) and not args.draw_zone:
        parser.error("Use --draw-zone to calibrate a different video.")

    capture = cv2.VideoCapture(str(args.video))
    try:
        frame, fps = read_first_frame(capture, args.video)
        if args.draw_zone:
            polygon = draw_zone(frame)
            if polygon is None:
                return
        else:
            corners = PARKING_ZONE if args.video.resolve() == PARKING_VIDEO.resolve() else DEFAULT_ZONE
            polygon = preset_zone(frame, corners)

        options = MonitorOptions(args.video, args.weights, args.device, args.initial, args.parking,
                                 args.database, args.output, EmailSettings.from_environment(), args.frames)
        started = time.perf_counter()
        tick = started
        with closing(process_video(capture, frame, fps, polygon, options)) as previews:
            for index, preview in enumerate(previews, 1):
                if not args.headless:
                    scale = min(1, 1100 / preview.frame.shape[1])
                    cv2.imshow(WINDOW, cv2.resize(preview.frame, None, fx=scale, fy=scale))
                    delay = max(1, round(1000 * (1 / fps - (time.perf_counter() - tick))))
                    if cv2.waitKey(delay) & 0xff in (27, ord("q")):
                        break
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                tick = time.perf_counter()
        elapsed = time.perf_counter() - started
        print(f"{index} frames at {index / elapsed:.1f} FPS | "
              f"{preview.entries} entries | {preview.exits} exits")
        print(f"Event log: {options.database}")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()


def read_first_frame(capture, path):
    ok, frame = capture.read()
    if not ok:
        raise ValueError(f"Cannot read video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Cannot determine the video frame rate.")
    return frame, fps


def process_video(capture, frame, fps, polygon, options, show_overlay=lambda: True):
    """Yield each processed frame. Close this iterator when stopping early."""
    from ultralytics import YOLO
    import torch

    model = YOLO(options.weights)
    device = options.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    half = torch.cuda.is_available() and device != "cpu"
    options.output.mkdir(parents=True, exist_ok=True)
    options.database.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    with EventLog(options.database, options.video, options.output, options.email) as database:
        try:
            yield from run(capture, frame, fps, polygon, model, options, database, run_id,
                           device, half, show_overlay)
        finally:
            capture.release()


def recover_track(tracks, active_ids, frame_number, point, distance, fps, width):
    """Reconnect a detection near the zone to one recently lost inside track."""
    if distance < -width * .035:
        return None
    candidates = []
    for tracker_id, track in tracks.items():
        gap = frame_number - track.last_frame
        if (tracker_id in active_ids or track.inside is not True
                or not 2 <= gap <= round(fps * 1.25) or len(track.trail) < 5):
            continue
        start = np.array(track.trail[0])
        last = np.array(track.trail[-1])
        velocity = (last - start) / (len(track.trail) - 1)
        movement = np.array(point) - last
        if np.dot(movement, velocity) <= 0:
            continue
        error = np.linalg.norm(movement - velocity * gap)
        tolerance = width * (.018 if distance >= 0 else .065)
        if error <= tolerance:
            candidates.append(tracker_id)
    if len(candidates) == 1:
        return tracks.pop(candidates[0])
    return None

def run(capture, frame, fps, polygon, model, options, database, run_id, device, half, show_overlay):
    reader = None
    if options.read_plates:
        from plate_reader import PlateReader, confirmed_plate
        reader = PlateReader()
    tracks = {}
    entries = exits = frame_number = 0
    latest_event = -float("inf")
    latest_message = ""
    initial = options.initial
    auto = initial is None
    width = frame.shape[1]
    while True:
        frame_number += 1
        video_seconds = (frame_number - 1) / fps
        results = model.track(frame, persist=True, tracker=str(ROOT / "highway_bytetrack.yaml"), conf=.1,
                              classes=VEHICLE_CLASSES, device=device, half=half, agnostic_nms=True, iou=.45, verbose=False)[0]
        boxes = results.boxes
        events = []
        visible_inside = 0
        source_frame = frame.copy() if reader else frame
        cv2.polylines(frame, [polygon], True, (0, 220, 255), 3)
        preview_frame = frame if show_overlay() else frame.copy()
        if boxes is not None and boxes.id is not None:
            active_ids = boxes.id.int().cpu().tolist()
            for coords, class_id, track_id in zip(boxes.xyxy.cpu().tolist(),
                                                   boxes.cls.int().cpu().tolist(), active_ids):
                x1, y1, x2, y2 = map(int, coords)
                point = ((x1 + x2) // 2, y2)
                distance = cv2.pointPolygonTest(polygon, point, True)
                if track_id not in tracks:
                    tracks[track_id] = (recover_track(tracks, active_ids, frame_number, point,
                                                      distance, fps, width) or Track(track_id))
                track = tracks[track_id]
                event = track.update(frame_number, point, distance, width * .003)
                if frame_number == 1:
                    track.inside = distance >= 0
                visible_inside += int(track.inside is True)
                track.class_votes[results.names[class_id]] += 1
                name = track.logged_class or track.class_votes.most_common(1)[0][0]
                if reader and track.plate is None and frame_number >= track.plate_sample_frame:
                    track.plate_sample_frame = frame_number + max(1, round(fps * .25))
                    crop = source_frame[max(0, y1):y2, max(0, x1):x2]
                    reading = reader.read(crop)
                    if reading:
                        track.plate_reads.append(reading)
                        track.plate = confirmed_plate(track.plate_reads)
                        if track.plate:
                            database.record_plate(run_id, track.vehicle_id, track.plate)
                if event:
                    track.logged_class = name
                    if event == "entry":
                        entries += 1
                    else:
                        exits += 1
                    latest_event = video_seconds
                    latest_message = f"{event.title()}: {name} #{track.vehicle_id}"
                    balance = initial + entries - exits
                    basis = "recovered track" if track.vehicle_id != track_id else "observed crossing"
                    events.append((event, track.vehicle_id, name, balance, basis))
                if not reader and not track.inside and event != "exit":
                    continue
                color = (70, 240, 70) if track.inside else (255, 195, 85)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, point, 4, color, -1)
                cv2.putText(frame, f"{name} #{track.vehicle_id}" + (f"  {track.plate}" if track.plate else ""), (x1, max(15, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
                if len(track.trail) >= 4:
                    cv2.arrowedLine(frame, track.trail[0], point, color, 2, tipLength=.25)
        if frame_number % 100 == 0:
            tracks = {key: track for key, track in tracks.items()
                      if frame_number - track.last_frame <= max(100, fps * 3)}

        if initial is None:
            initial = visible_inside
            print(f"Detected starting occupancy: {initial}")
        balance = initial + entries - exits
        status = (f"In {entries} | Out {exits} | Inside (by count) {max(0, balance)} "
                  f"| Visible {visible_inside} | Start {initial} | {video_seconds:.1f}s")
        if balance != visible_inside:
            status += " | Count differs from view"
        for canvas in (frame,) if frame is preview_frame else (frame, preview_frame):
            cv2.rectangle(canvas, (0, 0), (width, 42), (22, 22, 22), -1)
            cv2.putText(canvas, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        .62, (255, 255, 255), 2)
            if video_seconds - latest_event < 1.5:
                cv2.putText(canvas, latest_message, (12, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            .75, (35, 40, 255), 2)
        for event, track_id, name, event_balance, basis in events:
            screenshot = ""
            if event == "entry":
                snapshot = options.output / f"{run_id}_entry_{frame_number}_{track_id}.jpg"
                if cv2.imwrite(str(snapshot), frame):
                    screenshot = snapshot.name
                else:
                    print(f"Could not save {snapshot}")
            row = (event, round(video_seconds, 3), frame_number, track_id, name,
                   max(0, event_balance), event_balance < 0, screenshot, basis,
                   visible_inside, initial, "auto" if auto else "manual")
            database.record(row, run_id)
        yield Preview(preview_frame, entries, exits, max(0, balance), visible_inside,
                      video_seconds, fps, balance != visible_inside)
        if options.frame_limit is not None and frame_number >= options.frame_limit:
            break
        ok, frame = capture.read()
        if not ok:
            break


if __name__ == "__main__":
    main()

