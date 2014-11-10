import cv2

# This script is a quick test script to show the PointGrey Firefly image

# Change 0 to the index that works
cap = cv2.VideoCapture(0)

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        # Convert the camera from bayer format to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR);
        cv2.imshow('frame', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


