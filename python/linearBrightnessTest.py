import cv2
import math
import pandas as pd
import numpy as np
import time, sys, os, shutil
import yaml
from multiprocessing import Process, Queue
from Queue import Empty
import imageFeatures as imf

"""
# This script collects data
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

frames = pd.Series([], dtype=int, name='Frame')
data = pd.DataFrame(index=frames)

params = {}

def setParam(name, x):
    params[name] = x

print 'Run name:',
shortname = raw_input()

cv2.namedWindow('frame')

# Change 0 to the index that works
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

# Create the output directory and copy over stuff
for i in range(100):
    foldername = 'data/' + shortname + '_' + str(i)
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        break

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

# Helper variables
t = 0
i = 0

# This is for waiting for the autoexposure to settle
autoSettle = 0

if cap0.isOpened() and cap1.isOpened():
    q = Queue()
    p = Process(target=imageSaver, args=(foldername, q,))

    p.start()

    # Turn off white balance
    cap0.set(17, -4)
    cap0.set(26, -4)
    cap1.set(17, -4)
    cap1.set(26, -4)

    while True:
        if t < len(data) and i == 0:
            cap0.set(15, data.loc[t, 'Shutter 0'])
            cap0.set(14, data.loc[t, 'Gain 0'])
            cap1.set(15, data.loc[t, 'Shutter 1'])
            cap1.set(14, data.loc[t, 'Gain 1'])
            if abs(data.loc[t, 'Shutter 0']+2.0) < 0.1:
                autoSettle = 24
            else:
                autoSettle = 0

        i += 1

        if t >= len(data) or i-autoSettle > 6:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()

            if ret0 and ret1:
                frame0 = cv2.cvtColor(frame0, cv2.COLOR_BAYER_BG2BGR)
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BAYER_BG2BGR)
                disp = np.concatenate((frame0, frame1), axis=1)

                if t < len(data):
                    data.loc[t, 'Timestamp'] = currentTimestamp()

                    data.loc[t, 'Shutter 0'] = cap0.get(15)
                    data.loc[t, 'Gain 0'] = cap0.get(14)
                    imgname0 = shortname + '_0_{:0>4d}.png'.format(t)
                    data.loc[t, 'Image File 0'] = imgname0
                    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
                    data.loc[t, 'Mean Lum 0'] = imf.meanLuminance(gray0)
                    data.loc[t, 'Contrast 0'] = imf.contrast(gray0)
                    q.put((imgname0, frame0))

                    data.loc[t, 'Shutter 1'] = cap1.get(15)
                    data.loc[t, 'Gain 1'] = cap1.get(14)
                    imgname1 = shortname + '_1_{:0>4d}.png'.format(t)
                    data.loc[t, 'Image File 1'] = imgname1
                    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                    data.loc[t, 'Mean Lum 1'] = imf.meanLuminance(gray1)
                    data.loc[t, 'Contrast 1'] = imf.contrast(gray1)
                    q.put((imgname1, frame1))

                    t += 1

                cv2.imshow('frame', disp)

            i = 0

        else:
            cap0.grab()
            cap1.grab()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('w') and t >= len(data):
            t = len(data)
            shutterRange = np.linspace(1.0, 531.0, 24)
            gainRange = np.linspace(16.0, 64.0, 8)
            cap0.set(15, 1.0)
            cap0.set(14, 16.0)
            cap1.set(15, 1.0)
            cap1.set(14, 16.0)
            k = 0
            for s in shutterRange:
                for g in gainRange:
                    data.loc[t+k, 'Shutter 0'] = s
                    data.loc[t+k, 'Shutter 1'] = s
                    data.loc[t+k, 'Gain 0'] = g
                    data.loc[t+k, 'Gain 1'] = g
                    k += 1

            i = 0

q.put(False)

q.close()
p.join()

if len(data) > 0:
    data.to_csv(foldername + '/' + shortname + '_rawdata.csv')
