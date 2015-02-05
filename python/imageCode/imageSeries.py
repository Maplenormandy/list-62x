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

# fd feature detector. Note the high value is just for display
fd = cv2.SURF(400)

t = 0.0;

expo = 50
i = 0

if cap.isOpened():
    setCapExposure(expo)

    while True:
        ret, frame = cap.read()

        if ret:
            # Convert the camera from bayer format to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR);

            cv2.imshow('frame', frame)
            if i == 5:
                cv2.imwrite('./image_' + str(expo) + '.png', frame)

        if i < 5:
            i += 1
        else:
            i = 0
            expo += 50
            if expo < 500:
                setCapExposure(expo)
            else:
                break



        t += 0.030;

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


