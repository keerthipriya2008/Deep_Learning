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
        classes=[2, 3, 5, 7],
        verbose=False
    )

    vehicle_count = len(results[0].boxes)

    output = results[0].plot()

    cv2.putText(
        output,
        f"Vehicle Count: {vehicle_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Vehicle Counting", output)

    print("Vehicles detected:", vehicle_count)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()