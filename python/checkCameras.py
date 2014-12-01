import cv2
import math

# This script is a quick test script to show the PointGrey Firefly image

# Change 0 to the index that works
cap = cv2.VideoCapture(0)
cv2.namedWindow('frame')

# Create a trackbar for setting the exposure manually
def setCapExposure(x):
    cap.set(15, x)

def setCapGain(x):
    cap.set(14, x)

cv2.createTrackbar('Shutter', 'frame', 1, 531, setCapExposure)
cv2.createTrackbar('Gain', 'frame', 1, 531, setCapExposure)

# fd feature detector. Note the high value is just for display
fd = cv2.ORB()

t = 0.0;

if cap.isOpened():
    while True:
        ret, frame = cap.read()

        if ret:
            # Convert the camera from bayer format to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR);

            # Use fd feature detector
            kp = fd.detect(frame, None)
            disp = cv2.drawKeypoints(frame, kp, None, (255,0,0), 4)

            cv2.imshow('frame', disp)

        # Try setting the camera settings


        t += 0.030;

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


