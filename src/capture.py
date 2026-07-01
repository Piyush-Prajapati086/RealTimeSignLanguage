import cv2  #cv2 is the OpenCV library.

cap = cv2.VideoCapture(0)  #It opens the camera where 0 is first webcam, 1 is external webcam

while True:
    success, frame = cap.read()  #it asks "Camera, give me the next image.", Success return boolean and frame is the actual image

    if not success:
        print("Camera read failed")
        break

    cv2.imshow("Webcam", frame) #displays the image in real time 

    key = cv2.waitKey(1) #It waits for a keyboard press.

    print("Key =", key)

    if key == ord('s'): #ord() converts a character into its ASCII value.
        print("Saving...")
        cv2.imwrite("photo.jpg", frame) #writes the current frame to disk. It literally saves current frame. If you press S multiple times with the same filename, the previous photo.jpg will be overwritten.

    if key == ord('q'):
        print("Q Pressed")
        break

cap.release() #"I'm done with the webcam."
cv2.destroyAllWindows() #This closes every OpenCV window.