import csv
import os
import time
import cv2


CSV_HEADER = [
    "timestamp",
    "frame",
    "queue_size",
    "queue_length_px",
    "queue_length_m",
    "scene_label",
    "scene_confidence",
    "advisory_message",
]


class MetricsLogger:
    """
    Writes one CSV row per logged frame and optionally saves annotated video.

    CSV is flushed every LOG_EVERY_N_FRAMES frames so data is not lost
    if the program is interrupted.
    """

    LOG_EVERY_N_FRAMES = 15   # log to CSV every N frames (~0.5s at 30fps)

    def __init__(self,
                 csv_path="outputs/queue_log.csv",
                 video_path=None,
                 frame_size=(960, 540),
                 fps=25.0,
                 pixels_per_meter=90.0):

        self.csv_path        = csv_path
        self.pixels_per_meter = pixels_per_meter
        self._frame_count    = 0
        self._csv_file       = None
        self._writer         = None
        self._video_writer   = None

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer   = csv.DictWriter(self._csv_file, fieldnames=CSV_HEADER)
        self._writer.writeheader()
        print(f"Logging metrics to: {csv_path}")

        if video_path:
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                video_path, fourcc, fps, frame_size
            )
            print(f"Saving annotated video to: {video_path}")

    def log(self, frame_number, queue_size, queue_length_px,
            scene_label, scene_confidence, advisory_message):
        """
        Call once per frame. CSV write is throttled by LOG_EVERY_N_FRAMES.
        """
        self._frame_count += 1

        if self._frame_count % self.LOG_EVERY_N_FRAMES != 0:
            return

        queue_length_m = queue_length_px / self.pixels_per_meter

        self._writer.writerow({
            "timestamp":        time.strftime("%H:%M:%S"),
            "frame":            frame_number,
            "queue_size":       queue_size,
            "queue_length_px":  round(queue_length_px, 1),
            "queue_length_m":   round(queue_length_m, 2),
            "scene_label":      scene_label,
            "scene_confidence": round(scene_confidence, 3),
            "advisory_message": advisory_message,
        })
        self._csv_file.flush()

    def write_frame(self, frame):
        """Write an annotated frame to the output video (if video writer is open)."""
        if self._video_writer is not None:
            self._video_writer.write(frame)

    def close(self):
        if self._csv_file:
            self._csv_file.close()
        if self._video_writer:
            self._video_writer.release()
        print(f"Logger closed. CSV rows written to {self.csv_path}")
