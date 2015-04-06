import cv2
import math
import pandas as pd
import numpy as np
import time, sys, os, shutil
import yaml
from multiprocessing import Process, Queue
from Queue import Empty
import random
import imageFeatures as imf

"""
# This script collects data
if len(sys.argv) < 2:
    print "No configuration file specified"
    collectData = False
    config = None
else:
    collectData = True
    try:
        with open(sys.argv[1]) as f:
            config = yaml.load(f.read())
    except:
        print "Error:", sys.exc_info()[0]
        raise
"""

def currentTimestamp():
    return pd.Timestamp(time.time()*1000000000)

def imageSaver(foldername, q):
    while True:
        toSave = None
        try:
            toSave = q.get(True, 1)
        except Empty:
            pass

        if toSave != None:
            if toSave == False:
                print "Done"
                break

            name, frame = toSave
            cv2.imwrite(foldername + '/' + name, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print "Wrote", foldername + '/' + name

"""
if collectData:
    # Parse the configuration file
    if 'settingsFile' in config:
        rdf = pd.read_csv(config['settingsFile'])
        totalFrames = len(rdf)
        gains0 = rdf['Gain 0']
        shutters0 = rdf['Shutter 0']
        gains1 = rdf['Gain 1']
        shutters1 = rdf['Shutter 1']
        timestamps = pd.Series([currentTimestamp()] * totalFrames)
        features = pd.Series([0] * totalFrames)
        imageFiles0 = pd.Series([''] * totalFrames)
        imageFiles1 = pd.Series([''] * totalFrames)
        frames = rdf['Frame']
"""

frames = pd.Series([], dtype=int, name='Frame')
data = pd.DataFrame(index=frames)

params = {}

def setParam(name, x):
    params[name] = x

print 'Run name:',
shortname = raw_input()

cv2.namedWindow('frame')

while True:
    print 'Parameter name (empty to terminate):',
    name = raw_input()
    if name != '':
        params[name] = 0
        print 'max:',
        pmax = int(raw_input())

        cv2.createTrackbar(name, 'frame', 0, pmax, lambda x: setParam(name, x))
    else:
        break

# Change 0 to the index that works
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

# Create the output directory and copy over stuff
for i in range(100):
    foldername = 'data/' + shortname + '_' + str(i)
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        break

"""
shutil.copy(sys.argv[1], foldername)
if 'settingsFile' in config:
    shutil.copy(config['settingsFile'], foldername)
"""

def setCap0Exposure(x):
    cap0.set(15,x)
def setCap1Exposure(x):
    cap1.set(15,x)
def setCap0Gain(x):
    cap0.set(14,x)
def setCap1Gain(x):
    cap1.set(14,x)
def setCap0Auto(x):
    cap0.set(21,x)
def setCap1Auto(x):
    cap1.set(21,x)

def findSettings(oldSettings, oldFeatures, newFeatures):
    oldShutter, oldGain = oldSettings
    newShutter = 1.0
    newGain = 16.0

    oldMeanLum = oldFeatures
    newMeanLum = newFeatures

    oldExposure = imf.settingsToExposure(oldShutter, oldGain)

    newExposure = 111.2148 + 0.6940*oldExposure - 2.7011*oldMeanLum + 2.6972*newMeanLum
    newShutter, newGain = imf.exposureToSettings(newExposure)

    return newShutter, newGain

def usableMatch(matches, keypoints, keypointsBaseline):
	correctMatches = []
	minAmmount = 5
	srcPts=[]
	dstPts=[]
	for m,n in matches:
		if m.distance <.75*n.distance:
			correctMatches.append(m)
	if len(correctMatches)>minAmmount:
		dst_pts = np.float32([ keypoints[m.trainIdx].pt for m in correctMatches ])
		src_pts = np.float32([ keypointsBaseline[m.queryIdx].pt for m in correctMatches ])
		ransacMatches, mask= cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()

		matchesMask = np.array(matchesMask)
		numMatches = (matchesMask>.5).sum()
		efficiency = [numMatches, len(keypoints)]
	else:
		efficiency = [0, len(keypoints)]
	return efficiency

"""
if not collectData:
    cv2.createTrackbar('Shutter Baseline', 'frame', 1, 531, setCap0Exposure)
    cv2.createTrackbar('Gain Baseline', 'frame', 16, 64, setCap0Gain)
    cv2.createTrackbar('Shutter Compared', 'frame', 1, 531, setCap1Exposure)
    cv2.createTrackbar('Gain Compared', 'frame', 16, 64, setCap1Gain)
"""

# Helper variables
t = 0
i = 0
runNum = 0
startT = 0

expCam0 = True

writing = False
resetRun = False

index_params = dict(algorithm = 0, trees = 5)
search_params = dict(checks=50)

def surfDetectAndMatch(name, q, dq):
    surf = cv2.SURF()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    oldFrame = None

    oldKp = None
    oldDesc = None

    while True:
        newFrame = None
        try:
            newFrame = q.get(True, 1)
            print name + ": " + str(q.qsize()) + " left"
        except Empty:
            if oldFrame != None:
                print name + ": Resetting"
                oldFrame = None

        if newFrame != None:
            if newFrame == False:
                dq.close()
                print name + ": Done"
                break

            kp, desc = surf.detectAndCompute(newFrame[1], None)

            if oldFrame != None:
                if newFrame[0] == oldFrame[0]:
                    print name + ": New run detected"
                elif newFrame[0]-oldFrame[0] > 1:
                    print name + ": Warning, t mismatch!"

                succTrackFeatures = 0
                if desc != None and oldDesc != None:
                    matches = flann.knnMatch(oldDesc, desc, k=2)
                    efficiency = usableMatch(matches, kp, oldKp)
                    succTrackFeatures = efficiency[0]

                dq.put((newFrame[0], succTrackFeatures))



            oldFrame = newFrame
            oldKp = kp
            oldDesc = desc



if cap0.isOpened() and cap1.isOpened():
    q = Queue()
    p = Process(target=imageSaver, args=(foldername, q,))

    q0 = Queue()
    dq0 = Queue()
    p0 = Process(target=surfDetectAndMatch, args=("SDAM 0", q0, dq0,))
    q1 = Queue()
    dq1 = Queue()
    p1 = Process(target=surfDetectAndMatch, args=("SDAM 1", q1, dq1,))

    p.start()
    p0.start()
    p1.start()

    """
    if not collectData:
        cv2.setTrackbarPos('Shutter Baseline', 'frame', int(cap0.get(15)))
        cv2.setTrackbarPos('Gain Baseline', 'frame', int(cap0.get(14)))
        cv2.setTrackbarPos('Shutter Compared', 'frame', int(cap1.get(15)))
        cv2.setTrackbarPos('Gain Compared', 'frame', int(cap1.get(14)))
    """

    while True:
        i += 1

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if ret0 and ret1:
            frame0 = cv2.cvtColor(frame0, cv2.COLOR_BAYER_BG2BGR)
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BAYER_BG2BGR)

            disp = np.concatenate((frame0, frame1), axis=1)


            try:
                t0, succTrackFeatures0 = dq0.get_nowait()
                data.loc[t0, 'Succesfully Tracked Features 0'] = succTrackFeatures0
            except Empty:
                pass

            try:
                t1, succTrackFeatures1 = dq1.get_nowait()
                data.loc[t1, 'Succesfully Tracked Features 1'] = succTrackFeatures1
            except Empty:
                pass

            if writing and i > 6:
                gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

                q0.put((t, gray0))
                q1.put((t, gray1))

                # Calculate image features
                if expCam0:
                    meanLum = imf.meanLuminance(gray0)
                    camSettings = (cap0.get(15), cap0.get(14))
                else:
                    meanLum = imf.meanLuminance(gray1)
                    camSettings = (cap1.get(15), cap1.get(14))

                if oldGray0 != None:

                    # Save raw data
                    data.loc[t, 'Timestamp'] = currentTimestamp()
                    data.loc[t, 'Run Number'] = runNum
                    data.loc[t, 'Baseline'] = 1 if expCam0 else 0

                    data.loc[t, 'Experimental Mean Luminance'] = meanLum

                    data.loc[t, 'Shutter 0'] = cap0.get(15)
                    data.loc[t, 'Gain 0'] = cap0.get(14)

                    data.loc[t, 'Shutter 1'] = cap1.get(15)
                    data.loc[t, 'Gain 1'] = cap1.get(14)

                    imgname0 = shortname + '_0_{:0>4d}.png'.format(t)
                    data.loc[t, 'Image File 0'] = imgname0
                    imgname1 = shortname + '_1_{:0>4d}.png'.format(t)
                    data.loc[t, 'Image File 1'] = imgname1
                    q.put((imgname0, frame0))
                    q.put((imgname1, frame1))


                    newShutter, newGain = findSettings(oldCamSettings, oldMeanLum, meanLum)

                    # Determine new image settings
                    if expCam0:
                        cap0.set(14, newGain)
                        cap0.set(15, newShutter)
                    else:
                        cap1.set(14, newGain)
                        cap1.set(15, newShutter)

                    t += 1

                oldGray0 = gray0
                oldGray1 = gray1

                oldMeanLum = meanLum
                oldCamSettings = camSettings

                i = 0

            cv2.putText(disp, "Frame: " + str(t-startT), (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
            cv2.putText(disp, "Baseline: " + ("1" if expCam0 else "0"), (50,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
            cv2.imshow('frame', disp)


        else:
            cap0.grab()
            cap1.grab()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        # The order is to press 'w' when starting a run, then press 'r' to do it again in a pair
        elif key == ord('w'):
            expCam0 = random.choice((True, False))
            resetRun = True
        elif key == ord('r'):
            expCam0 = not expCam0
            resetRun = True
        elif key == ord('s'):
            writing = False
            runNum += 1

        if resetRun:
            resetRun = False
            writing = True
            startT = t
            oldGray0 = None
            oldGray1 = None
            i = 0

            # To start off, set auto-exposure
            cap0.set(14, -2)
            cap0.set(15, -2)

            cap1.set(14, -2)
            cap1.set(15, -2)

q.put(False)
q0.put(False)
q1.put(False)

q.close()
dq0.close()
dq1.close()
q0.close()
q1.close()
p.join()
p0.join()
p1.join()

if len(data) > 0:
    data.to_csv(foldername + '/' + shortname + '_rawdata.csv')
