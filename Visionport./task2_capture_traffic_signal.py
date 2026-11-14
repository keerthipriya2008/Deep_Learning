import cv2
import os

os.makedirs("captured_images", exist_ok=True)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not available")
    exit()

print("Camera started")
print("Press S to capture image")
print("Press Q to quit")

while True:

    ret, frame = camera.read()

    if not ret:
        print("Unable to capture image")
        break

    cv2.imshow("Traffic Signal Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):

        cv2.imwrite(
            "captured_images/traffic_signal.jpg",
            frame
        )

        print("Image captured successfully")

    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()