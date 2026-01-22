import cv2
import os

def extract_frames(video_path, out_dir, every_n_frames=30):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count, saved = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n_frames == 0:
            cv2.imwrite(f"{out_dir}/frame_{saved}.jpg", frame)
            saved += 1
        count += 1

    cap.release()
