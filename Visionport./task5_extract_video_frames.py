import cv2
import os

video = cv2.VideoCapture("traffic_sample.mp4")

os.makedirs("extracted_frames", exist_ok=True)

frame_count = 0
saved_count = 0

while True:

    ret, frame = video.read()

    if not ret:
        break

    if frame_count % 30 == 0:

        filename = f"extracted_frames/frame_{saved_count}.jpg"

        cv2.imwrite(filename, frame)

        saved_count += 1

    frame_count += 1

video.release()

print("Total frames processed:", frame_count)
print("Frames saved:", saved_count)
print("Frame extraction completed")