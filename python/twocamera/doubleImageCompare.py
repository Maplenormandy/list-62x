import cv2
import math
import time
import numpy as np
from multiprocessing import Process, Queue
from Queue import Empty
import random


# This script is a quick test script to show the PointGrey Firefly image

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cv2.namedWindow('frame')

# Create a trackbar for setting the exposure manually
def setCapExposure(x):
    cap0.set(15, x)

def setCapGain(x):
    cap0.set(14, x)

# fd feature detector. Note the high value is just for display
fd = cv2.SURF(400)

t = 0.0;

def surfDetectAndMatch(q, dq):
    surf = cv2.SURF(400)

    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)

    while True:
        try:
            frames = q.get(True, 1)
            print q.qsize()
        except Empty:
            dq.close()
            print "Queue was empty"
            break

        disp = frames[0]
        gray0 = frames[1]
        gray1 = frames[2]


        kp0, desc0 = surf.detectAndCompute(gray0, None)
        kp1, desc1 = surf.detectAndCompute(gray1, None)

        h, w = gray0.shape[:2]

        matches = flann.knnMatch(desc0, desc1, k=2)
        matchesMask = [[0,0] for i in xrange(len(matches))]

        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                color = (0, 255, 0)
                matchesMask[i]=[1,0]
                cv2.line(disp, (int(kp0[m.queryIdx].pt[0]), int(kp0[m.queryIdx].pt[1])) , (int(kp1[m.trainIdx].pt[0] + w), int(kp1[m.trainIdx].pt[1])), color)

        dq.put(disp)

    print "Done"

if cap0.isOpened():
    q = Queue()
    dq = Queue()
    p = Process(target=surfDetectAndMatch, args=(q,dq,))
    p.start()

    i = 0

    while True:
        print 1.0/(time.time() - t)
        t = time.time()

        if q.qsize() == 0:
            ret, frame = cap0.read()
            ret2, frame2 = cap1.read()

            if ret and ret2:
                #print expo
                #cv2.setTrackbarPos('Shutter', 'frame', expo)
                # Convert the camera from bayer format to RGB

                frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BAYER_BG2BGR)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                dispf = np.concatenate((frame, frame2), axis=1)

                q.put((dispf, gray, gray2))
        else:
            cap0.grab()
            cap1.grab()

        try:
            disp = dq.get_nowait()
            cv2.imshow('frame', disp)
        except Empty:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    q.close()
    dq.close()
    p.join()
