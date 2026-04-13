from ultralytics import YOLO
import cv2
import numpy as np
import supervision as sv

from queue_detection.hybrid_queue import compute_hybrid_queue

VIDEO_PATH = "data/videos/test_queue.mp4"
SMOOTH_WINDOW = 5

def main():

    queue_sizes = []
    queue_lengths = []

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(VIDEO_PATH)

    tracker = sv.ByteTrack()

    # 🔥 Track history (position smoothing)
    track_history = {}
    MAX_HISTORY = 5

    # 🔥 NEW: queue membership memory (FIXES YOUR ISSUE)
    queue_history = {}
    QUEUE_MEMORY = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))

        # ==============================
        # STEP 1: YOLO detection
        # ==============================
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        mask = detections.class_id == 0
        detections = detections[mask]

        # ==============================
        # STEP 2: BYTE TRACK
        # ==============================
        detections = tracker.update_with_detections(detections)

        person_data = []  # (x, y, id)
        active_ids = set()

        # ==============================
        # STEP 3: Track + smoothing
        # ==============================
        for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):

            if track_id is None:
                continue

            active_ids.add(track_id)

            x1, y1, x2, y2 = map(int, xyxy)

            cx = int((x1 + x2) / 2)
            cy = int(y2)

            # update history
            if track_id not in track_history:
                track_history[track_id] = []

            track_history[track_id].append((cx, cy))

            if len(track_history[track_id]) > MAX_HISTORY:
                track_history[track_id].pop(0)

            # averaged position
            avg_x = int(np.mean([p[0] for p in track_history[track_id]]))
            avg_y = int(np.mean([p[1] for p in track_history[track_id]]))

            person_data.append((avg_x, avg_y, track_id))

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            cv2.circle(frame, (avg_x, avg_y), 5, (0, 0, 255), -1)

        # clean old tracks
        track_history = {
            tid: track_history[tid]
            for tid in active_ids if tid in track_history
        }

        # ==============================
        # STEP 4: Queue Detection
        # ==============================
        if len(person_data) >= 4:

            positions = [(x, y) for (x, y, _) in person_data]

            queue_info = compute_hybrid_queue(positions)

            if queue_info is not None:

                queue_points, _, queue_length = queue_info

                #  map queue points → IDs
                current_queue_ids = set()

                for qp in queue_points:
                    for (x, y, tid) in person_data:
                        if np.linalg.norm(np.array([x, y]) - qp) < 15:
                            current_queue_ids.add(tid)

                # ==============================
                #  STEP 5: TEMPORAL QUEUE MEMORY
                # ==============================
                for tid in current_queue_ids:
                    if tid not in queue_history:
                        queue_history[tid] = 0
                    queue_history[tid] += 1

                # decay
                for tid in list(queue_history.keys()):
                    if tid not in current_queue_ids:
                        queue_history[tid] -= 1

                    if queue_history[tid] <= 0:
                        del queue_history[tid]

                # stable queue IDs
                stable_queue_ids = [
                    tid for tid, v in queue_history.items() if v >= 2
                ]

                queue_size = len(stable_queue_ids)

                # draw queue points
                for p in queue_points:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 8, (255, 0, 255), -1)

                # smoothing
                queue_sizes.append(queue_size)
                queue_lengths.append(queue_length)

                queue_sizes = queue_sizes[-SMOOTH_WINDOW:]
                queue_lengths = queue_lengths[-SMOOTH_WINDOW:]

                smooth_size = int(sum(queue_sizes) / len(queue_sizes))
                smooth_length = int(sum(queue_lengths) / len(queue_lengths))

                # draw line
                queue_points = np.array(queue_points)
                start = queue_points[0]
                end = queue_points[-1]

                cv2.line(frame,
                         (int(start[0]), int(start[1])),
                         (int(end[0]), int(end[1])),
                         (255, 0, 0), 3)

                # display
                cv2.putText(frame, f"Queue Size: {smooth_size}",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,255), 2)

                cv2.putText(frame, f"Queue Length: {int(smooth_length)} px",
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,255), 2)

        cv2.imshow("Person Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()