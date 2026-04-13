# This file contains utility functions for video processing, such as playing a video.
import cv2


def play_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # resize frame
        frame = cv2.resize(frame, (960, 540))

        cv2.imshow("Queue Video Test", frame)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()