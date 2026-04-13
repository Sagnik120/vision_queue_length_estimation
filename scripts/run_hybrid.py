"""
Ablation Config A — Hybrid geometric baseline.

Run from project root:
    python scripts/run_hybrid.py
"""

import sys
import os

# allow imports from project root regardless of where the script is called from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from queue_detection.hybrid_queue import compute_hybrid_queue
from utils.metrics_logger import MetricsLogger

VIDEO_PATH        = "data/videos/test_queue.mp4"
OUTPUT_CSV        = "outputs/ablation_A.csv"
OUTPUT_VIDEO      = "outputs/ablation_A.mp4"
SMOOTH_WINDOW     = 5
MAX_TRACK_HISTORY = 5


def main():
    model = YOLO("yolov8n.pt")
    cap   = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"ERROR: cannot open {VIDEO_PATH}")
        return

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_size = (960, 540)

    tracker = sv.ByteTrack()
    logger  = MetricsLogger(
        csv_path=OUTPUT_CSV,
        video_path=OUTPUT_VIDEO,
        frame_size=frame_size,
        fps=fps,
    )

    track_history = {}
    queue_history = {}
    queue_sizes   = []
    queue_lengths = []
    frame_count   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, frame_size)
        frame_count += 1

        results    = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        mask       = detections.class_id == 0
        detections = detections[mask]
        detections = tracker.update_with_detections(detections)

        person_data = []
        active_ids  = set()

        for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):
            if track_id is None:
                continue

            active_ids.add(track_id)
            x1, y1, x2, y2 = map(int, xyxy)
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((cx, cy))
            if len(track_history[track_id]) > MAX_TRACK_HISTORY:
                track_history[track_id].pop(0)

            avg_x = int(np.mean([p[0] for p in track_history[track_id]]))
            avg_y = int(np.mean([p[1] for p in track_history[track_id]]))

            person_data.append((avg_x, avg_y, track_id))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 100), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 255, 0), 1)

        track_history = {t: v for t, v in track_history.items()
                         if t in active_ids}

        smooth_size   = 0
        smooth_length = 0.0

        if len(person_data) >= 4:
            positions  = [(x, y) for (x, y, _) in person_data]
            queue_info = compute_hybrid_queue(positions)

            if queue_info is not None:
                queue_points, _, queue_length = queue_info

                current_queue_ids = set()
                for qp in queue_points:
                    for (x, y, tid) in person_data:
                        if np.linalg.norm(np.array([x, y]) - qp) < 15:
                            current_queue_ids.add(tid)

                for tid in current_queue_ids:
                    queue_history[tid] = queue_history.get(tid, 0) + 1

                for tid in list(queue_history.keys()):
                    if tid not in current_queue_ids:
                        queue_history[tid] -= 1
                    if queue_history[tid] <= 0:
                        del queue_history[tid]

                stable_ids = [t for t, v in queue_history.items() if v >= 2]
                queue_sizes.append(len(stable_ids))
                queue_lengths.append(queue_length)
                queue_sizes   = queue_sizes[-SMOOTH_WINDOW:]
                queue_lengths = queue_lengths[-SMOOTH_WINDOW:]

                smooth_size   = int(sum(queue_sizes)   / len(queue_sizes))
                smooth_length = float(sum(queue_lengths) / len(queue_lengths))

                queue_pts = np.array(queue_points)
                cv2.line(frame,
                         tuple(map(int, queue_pts[0])),
                         tuple(map(int, queue_pts[-1])),
                         (255, 80, 0), 3)
                for p in queue_pts:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 7, (255, 0, 200), -1)

        cv2.putText(frame, f"[BASELINE] Queue Size: {smooth_size}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Queue Length: {int(smooth_length)} px",
                    (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (750, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        logger.log(
            frame_number=frame_count,
            queue_size=smooth_size,
            queue_length_px=smooth_length,
            scene_label="n/a",
            scene_confidence=0.0,
            advisory_message="baseline",
        )
        logger.write_frame(frame)

        cv2.imshow("Ablation A - Hybrid Baseline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.close()
    print(f"Done. Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
