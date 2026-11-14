import cv2

image_path = "captured_images/traffic_signal.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Image not found")
else:

    print("Image loaded successfully")

    height, width, channels = image.shape

    print("Image Width:", width)
    print("Image Height:", height)

    cv2.imshow("VisionPort Traffic Image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    