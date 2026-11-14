import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture("traffic_sample.mp4")

while video.isOpened():

    ret, frame = video.read()

    if not ret:
        break

    results = model(
        frame,
        classes=[2],
        verbose=False
    )

    output = results[0].plot()

    cv2.imshow("Car Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()