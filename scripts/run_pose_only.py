"""
Ablation Config B — Pose graph only (no smoother, no classifier).

FIX: replaced extract_keypoints(results, det_idx) with build_keypoint_map
+ lookup_keypoints pattern. After sv.Detections filtering + ByteTrack,
det_idx no longer matches results.keypoints.data indices — keypoints were
being assigned to wrong people. Now uses bbox-based nearest lookup.

Run from project root:
    python scripts/run_pose_only.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

from queue_detection.pose_graph import compute_pose_queue, compute_facing_angle
from utils.metrics_logger import MetricsLogger

VIDEO_PATH        = "data/videos/test_queue.mp4"
OUTPUT_CSV        = "outputs/ablation_B.csv"
OUTPUT_VIDEO      = "outputs/ablation_B.mp4"
MAX_TRACK_HISTORY = 5
SMOOTH_WINDOW     = 5


def build_keypoint_map(results):
    """
    Build bbox->keypoints lookup BEFORE any filtering.
    Keys are rounded int bbox coords — stable across filter/track steps.
    """
    kp_map = {}
    try:
        kps   = results.keypoints.data
        boxes = results.boxes.xyxy.cpu().numpy()
        for i in range(len(boxes)):
            key = tuple(map(int, boxes[i]))
            kp_map[key] = kps[i].cpu().numpy()
    except Exception:
        pass
    return kp_map


def lookup_keypoints(kp_map, xyxy):
    """Return keypoints for a tracked bbox via nearest-bbox lookup."""
    if not kp_map:
        return None
    target = np.array(xyxy, dtype=float)
    best_key, best_dist = None, float('inf')
    for key in kp_map:
        d = float(np.linalg.norm(target - np.array(key)))
        if d < best_dist:
            best_dist = d
            best_key  = key
    return kp_map[best_key] if best_key is not None and best_dist < 20 else None


def main():
    model = YOLO("yolov8n-pose.pt")
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

    track_history  = {}
    keypoint_store = {}
    queue_sizes    = []
    queue_lengths  = []
    frame_count    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, frame_size)
        frame_count += 1

        results = model(frame, verbose=False)[0]

        # FIX: build keypoint map BEFORE any filtering
        kp_map = build_keypoint_map(results)

        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]
        detections = tracker.update_with_detections(detections)

        person_data = []
        active_ids  = set()

        # FIX: removed det_idx — no longer needed
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

            hist  = track_history[track_id]
            avg_x = int(np.mean([p[0] for p in hist]))
            avg_y = int(np.mean([p[1] for p in hist]))
            velocity = ((hist[-1][0] - hist[-2][0],
                         hist[-1][1] - hist[-2][1])
                        if len(hist) >= 2 else (0, 0))

            # FIX: use bbox lookup instead of broken det_idx
            kp = lookup_keypoints(kp_map, xyxy)
            if kp is not None:
                keypoint_store[track_id] = kp

            angle = None
            if track_id in keypoint_store:
                angle = compute_facing_angle(keypoint_store[track_id])

            person_data.append({
                'id':        track_id,
                'pos':       (avg_x, avg_y),
                'angle':     angle,
                'velocity':  velocity,
                'keypoints': keypoint_store.get(track_id),
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

        track_history  = {t: v for t, v in track_history.items()  if t in active_ids}
        keypoint_store = {t: v for t, v in keypoint_store.items() if t in active_ids}

        smooth_size   = 0
        smooth_length = 0.0

        if len(person_data) >= 3:
            queue_result = compute_pose_queue(person_data, min_queue_size=2)

            if queue_result is not None:
                queue_persons = queue_result['queue_persons']
                queue_sizes.append(len(queue_persons))
                queue_lengths.append(queue_result['queue_length_px'])
                queue_sizes   = queue_sizes[-SMOOTH_WINDOW:]
                queue_lengths = queue_lengths[-SMOOTH_WINDOW:]

                smooth_size   = int(sum(queue_sizes)   / len(queue_sizes))
                smooth_length = float(sum(queue_lengths) / len(queue_lengths))

                if len(queue_persons) >= 2:
                    cv2.line(frame,
                             tuple(map(int, queue_persons[0]['pos'])),
                             tuple(map(int, queue_persons[-1]['pos'])),
                             (255, 80, 0), 3)

                for p in queue_persons:
                    cv2.circle(frame,
                               tuple(map(int, p['pos'])), 8,
                               (200, 80, 255), -1)

        cv2.putText(frame, f"[POSE ONLY] Queue Size: {smooth_size}",
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
            advisory_message="pose_only",
        )
        logger.write_frame(frame)

        cv2.imshow("Ablation B - Pose Graph Only", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.close()
    print(f"Done. Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()