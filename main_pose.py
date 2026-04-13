import cv2
import numpy as np
import networkx as nx
from ultralytics import YOLO
import supervision as sv

from queue_detection.pose_graph import compute_pose_queue, compute_facing_angle
from queue_detection.temporal_smoother import TemporalSmoother
from queue_detection.crowd_classifier import CrowdClassifier
from utils.langchain_alerts import QueueAdvisoryAgent
from utils.metrics_logger import MetricsLogger

VIDEO_PATH        = "data/videos/test_queue.mp4"
OUTPUT_VIDEO      = "outputs/queue_output.mp4"
OUTPUT_CSV        = "outputs/queue_log.csv"
SMOOTH_WINDOW     = 5
MAX_TRACK_HISTORY = 5

LABEL_COLOURS = {
    "queue":     (0, 220, 120),
    "crowd":     (0, 100, 255),
    "uncertain": (180, 180, 180),
}


def build_keypoint_map(results):
    """
    Build bbox->keypoints lookup BEFORE any filtering.

    After sv.Detections.from_ultralytics + class mask + ByteTrack,
    the loop index no longer matches results.keypoints.data order.
    This map uses rounded bbox coords as keys so we can do a
    nearest-bbox lookup during the tracking loop.
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
    target   = np.array(xyxy, dtype=float)
    best_key, best_dist = None, float('inf')
    for key in kp_map:
        d = float(np.linalg.norm(target - np.array(key)))
        if d < best_dist:
            best_dist = d
            best_key  = key
    return kp_map[best_key] if best_key is not None and best_dist < 20 else None


def get_graph_connected_ids(G, person_data):
    """
    Return the set of track IDs whose graph node belongs to any
    connected component of size >= 2.

    FIX: previously only MST spine members were fed to the smoother,
    so people adjacent to the queue but not on the longest path never
    had their EMA score raised and were never counted. Now everyone
    connected in the graph gets their score raised.
    """
    graph_node_indices = set()
    for component in nx.connected_components(G):
        if len(component) >= 2:
            graph_node_indices.update(component)

    return {person_data[i]['id']
            for i in graph_node_indices
            if i < len(person_data)}


def draw_advisory_banner(frame, advisory_text, scene_label):
    if not advisory_text:
        return
    h, w    = frame.shape[:2]
    colour  = LABEL_COLOURS.get(scene_label, (200, 200, 200))
    overlay = frame.copy()
    if len(advisory_text) > 90:
        split = advisory_text.rfind(" ", 0, 90)
        line1 = advisory_text[:split]
        line2 = advisory_text[split + 1:]
        cv2.rectangle(overlay, (0, h - 58), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, line1, (10, h - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, colour, 1)
        cv2.putText(frame, line2, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, colour, 1)
    else:
        cv2.rectangle(overlay, (0, h - 36), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, advisory_text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1)


def draw_density_heatmap(frame, person_data):
    if not person_data:
        return
    heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
    for p in person_data:
        cx, cy = map(int, p['pos'])
        heatmap[max(0, cy - 35):min(frame.shape[0], cy + 35),
                max(0, cx - 35):min(frame.shape[1], cx + 35)] += 1.0
    heatmap = cv2.GaussianBlur(heatmap, (71, 71), 0)
    thr     = heatmap.max() * 0.20
    mask    = heatmap > thr
    if not mask.any():
        return
    norm    = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr     = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    blended = frame.copy()
    cv2.addWeighted(bgr, 0.35, blended, 0.65, 0, blended)
    frame[mask] = blended[mask]


def draw_overlay(frame, queue_result, smoother, person_data,
                 classifier_label, classifier_conf,
                 smooth_size, smooth_length, stable_ids):

    draw_density_heatmap(frame, person_data)

    if queue_result is None:
        cv2.putText(frame, "No queue detected",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (180, 180, 180), 2)
        return

    G             = queue_result['graph']
    all_positions = queue_result['all_positions']
    queue_persons = queue_result['queue_persons']

    # draw graph edges
    for u, v, data in G.edges(data=True):
        pu = tuple(map(int, all_positions[u]))
        pv = tuple(map(int, all_positions[v]))
        cv2.line(frame, pu, pv,
                 (0, int(min(255, data['weight'] * 400)), 0), 1)

    # draw queue spine
    if len(queue_persons) >= 2:
        cv2.line(frame,
                 tuple(map(int, queue_persons[0]['pos'])),
                 tuple(map(int, queue_persons[-1]['pos'])),
                 (255, 80, 0), 3)

    # draw per-person confidence circles
    stable_set = set(stable_ids)
    for p in person_data:
        conf = smoother.get_score(p['id'])
        if conf < 0.10:
            continue
        cx, cy = map(int, p['pos'])
        colour = (80, int(conf * 220), 200)
        if p['id'] in stable_set:
            cv2.circle(frame, (cx, cy), 10, colour, -1)
        else:
            cv2.circle(frame, (cx, cy), 7, colour, 2)
        cv2.putText(frame, f"{conf:.2f}", (cx - 14, cy - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, colour, 1)

    # HUD
    cv2.putText(frame, f"Queue Size: {smooth_size}",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, f"Queue Length: {int(smooth_length)} px",
                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    label_colour = LABEL_COLOURS.get(classifier_label, (200, 200, 200))
    cv2.putText(frame,
                f"Scene: {classifier_label.upper()}  ({classifier_conf:.0%})",
                (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, label_colour, 2)


def run(video_path=None):
    """
    Main processing function.
    Accepts optional video_path so Streamlit can call run(video_path="...")
    without monkeypatching module globals.
    """
    vpath = video_path or VIDEO_PATH

    model = YOLO("yolov8n-pose.pt")
    cap   = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"ERROR: cannot open {vpath}")
        return

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_size = (960, 540)

    tracker    = sv.ByteTrack()

    # FIX: lower stable_threshold to 0.45 (was 0.55)
    # 0.55 required ~4 consecutive frames to confirm — too slow for a
    # person who joins the queue and stands at the edge of the MST spine.
    # 0.45 requires ~3 frames, which is still temporally stable but
    # responsive enough to count edge members correctly.
    smoother   = TemporalSmoother(stable_threshold=0.45)

    classifier = CrowdClassifier(history_len=15)
    agent      = QueueAdvisoryAgent(location="Main Entrance")
    logger     = MetricsLogger(
        csv_path=OUTPUT_CSV,
        video_path=OUTPUT_VIDEO,
        frame_size=frame_size,
        fps=fps,
    )

    track_history  = {}
    keypoint_store = {}
    queue_sizes    = []
    queue_lengths  = []
    last_label     = "uncertain"
    last_conf      = 0.0
    last_advisory  = ""
    frame_count    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, frame_size)
        frame_count += 1

        results = model(frame, verbose=False)[0]

        # build keypoint map BEFORE any filtering or tracking
        kp_map = build_keypoint_map(results)

        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 0]
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

            hist     = track_history[track_id]
            avg_x    = int(np.mean([p[0] for p in hist]))
            avg_y    = int(np.mean([p[1] for p in hist]))
            velocity = ((hist[-1][0] - hist[-2][0],
                         hist[-1][1] - hist[-2][1])
                        if len(hist) >= 2 else (0, 0))

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
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            if angle is not None:
                ax = int(avg_x + 28 * np.sin(angle))
                ay = int(avg_y - 28 * np.cos(angle))
                cv2.arrowedLine(frame, (avg_x, avg_y), (ax, ay),
                                (0, 200, 255), 2, tipLength=0.4)

        track_history  = {t: v for t, v in track_history.items()
                          if t in active_ids}
        keypoint_store = {t: v for t, v in keypoint_store.items()
                          if t in active_ids}

        queue_result  = None
        smooth_size   = 0
        smooth_length = 0.0
        stable_ids    = []

        if len(person_data) >= 3:
            queue_result = compute_pose_queue(person_data, min_queue_size=2)

            if queue_result is not None:
                G = queue_result['graph']

                # FIX: feed ALL graph-connected people to the smoother,
                # not just the MST spine (queue_persons).
                #
                # Previously:
                #   current_ids = {p['id'] for p in queue_result['queue_persons']}
                # This meant people adjacent to the queue but not on the
                # longest MST path never had their EMA score raised, so
                # they were never counted even if clearly queueing.
                #
                # Now: anyone whose graph node is in a connected component
                # of size >= 2 gets their score raised each frame they
                # remain connected. The MST spine is still used for
                # ordering and length measurement — not for membership.
                graph_ids = get_graph_connected_ids(G, person_data)
                smoother.update(graph_ids)
                stable_ids = smoother.get_stable_ids()

                # only count stable members who are still graph-connected
                # this prevents stale decayed IDs from inflating the count
                stable_ids = [tid for tid in stable_ids
                              if tid in graph_ids]

                queue_sizes.append(len(stable_ids))
                queue_lengths.append(queue_result['queue_length_px'])
                queue_sizes   = queue_sizes[-SMOOTH_WINDOW:]
                queue_lengths = queue_lengths[-SMOOTH_WINDOW:]
                smooth_size   = int(sum(queue_sizes)   / len(queue_sizes))
                smooth_length = float(sum(queue_lengths) / len(queue_lengths))

                last_label, last_conf, _ = classifier.update(
                    person_data, G)
                last_advisory = agent.get_advisory(
                    smooth_size, smooth_length, last_label, last_conf)

                if last_advisory != getattr(run, '_prev_advisory', ''):
                    print(f"[Frame {frame_count}] {last_advisory}")
                    run._prev_advisory = last_advisory

            else:
                smoother.update(set())

        draw_overlay(frame, queue_result, smoother, person_data,
                     last_label, last_conf, smooth_size, smooth_length,
                     stable_ids)
        draw_advisory_banner(frame, last_advisory, last_label)
        cv2.putText(frame, f"Frame: {frame_count}", (750, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        logger.log(frame_count, smooth_size, smooth_length,
                   last_label, last_conf, last_advisory)
        logger.write_frame(frame)

        cv2.imshow("Queue Estimation - Pose Aware", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.close()
    print(f"Done. {frame_count} frames.")
    print(f"Video : {OUTPUT_VIDEO}")
    print(f"CSV   : {OUTPUT_CSV}")


def main():
    run()


if __name__ == "__main__":
    main()