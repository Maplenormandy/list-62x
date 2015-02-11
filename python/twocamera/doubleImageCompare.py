import cv2
import math
import time
import numpy as np
from multiprocessing import Process, Queue
from Queue import Empty
import random
from lsi import intersection


# This script is a quick test script to show the PointGrey Firefly image

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cv2.namedWindow('frame')

# fd feature detector. Note the high value is just for display
fd = cv2.SURF(400)

t = 0.0;

def setCap0Exposure(x):
    cap0.set(15,x)
def setCap1Exposure(x):
    cap1.set(15,x)
def setCap0Gain(x):
    cap0.set(14,x)
def setCap1Gain(x):
    cap1.set(14,x)

cv2.createTrackbar('Shutter Baseline', 'frame', 1, 531, setCap0Exposure)
cv2.createTrackbar('Gain Baseline', 'frame', 16, 64, setCap0Gain)
cv2.createTrackbar('Shutter Compared', 'frame', 1, 531, setCap1Exposure)
cv2.createTrackbar('Gain Compared', 'frame', 16, 64, setCap1Gain)

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

        if desc0 != None and desc1 != None:
            matches = flann.knnMatch(desc0, desc1, k=2)

            matchesMask = [[0,0] for i in xrange(len(matches))]

            # This is used as a heuristic for how many bad matches there are, based on the intuition that mismatched feature lines cross
            listOfLines = []

            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    color = (0, 255, 0)
                    matchesMask[i]=[1,0]
                    kp0Pt = kp0[m.queryIdx].pt
                    kp1Pt = kp1[m.trainIdx].pt
                    cv2.line(disp, (int(kp0Pt[0]), int(kp0Pt[1])) , (int(kp1Pt[0] + w), int(kp1Pt[1])), color)
                    # The implementation I'm using apparently "doesn't handle horizontal and vertical lines"
                    listOfLines.append(((kp0Pt[0],kp0Pt[1]+kp0Pt[0]),(kp1Pt[0]+w,kp1Pt[1]+kp1Pt[0]+w)))

            if len(listOfLines) > 0:
                featureIntersections = intersection(listOfLines)
                dq.put((disp, len(featureIntersections)))
            else:
                dq.put((disp, 0))
        else:
            dq.put((disp, 0))


    print "Done"

if cap0.isOpened():
    q = Queue()
    dq = Queue()
    p = Process(target=surfDetectAndMatch, args=(q,dq,))
    p.start()

    cv2.setTrackbarPos('Shutter Baseline', 'frame', int(cap0.get(15)))
    cv2.setTrackbarPos('Gain Baseline', 'frame', int(cap0.get(14)))
    cv2.setTrackbarPos('Shutter Compared', 'frame', int(cap1.get(15)))
    cv2.setTrackbarPos('Gain Compared', 'frame', int(cap1.get(14)))

    i = 4

    while True:
        print 1.0/(time.time() - t)
        t = time.time()

        i += 1

        if q.qsize() == 0 and i > 3:
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

            i = 0
        else:
            cap0.grab()
            cap1.grab()

        try:
            disp, numIntersections = dq.get_nowait()
            cv2.putText(disp, "Intersections: " + str(numIntersections), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            cv2.imshow('frame', disp)
        except Empty:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    q.close()
    dq.close()
    p.join()
