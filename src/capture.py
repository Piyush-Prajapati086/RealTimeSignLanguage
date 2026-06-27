import cv2

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        print("Camera read failed")
        break

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    print("Key =", key)

    if key == ord('s'):
        print("Saving...")
        cv2.imwrite("photo.jpg", frame)

    if key == ord('q'):
        print("Q Pressed")
        break

cap.release()
cv2.destroyAllWindows()