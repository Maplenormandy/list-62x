import cv2
import math
import pandas as pd
import numpy as np
import time, sys, os, shutil
import yaml

# This script collects data
if len(sys.argv) < 2:
    sys.exit("Please specify a configuration file")

config = None

try:
    with open(sys.argv[1]) as f:
        config = yaml.load(f.read())
except:
    print "Error:", sys.exc_info()[0]
    raise

def currentTimestamp():
    return pd.Timestamp(time.time()*1000000000)

# Parse the configuration file
if 'settingsFile' in config:
    rdf = pd.read_csv(config['settingsFile'])
    totalFrames = len(rdf)
    gains = rdf['Gain']
    shutters = rdf['Shutter']
    timestamps = pd.Series([currentTimestamp()] * totalFrames)
    features = pd.Series([0] * totalFrames)
    imageFiles = pd.Series([''] * totalFrames)
    frames = rdf['Frame']


# Change 0 to the index that works
cap = cv2.VideoCapture(0)
cv2.namedWindow('frame')

# fd feature detector. Note the high value is just for display
fd = cv2.ORB(nfeatures=2000)

t = 0

if cap.isOpened():
    cap.set(15, shutters[0])
    cap.set(14, gains[0])

    while True:
        ret, frame = cap.read()

        if ret:
            # Convert the camera from bayer format to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)

            kp = fd.detect(frame, None)
            disp = cv2.drawKeypoints(frame, kp, None, (255,0,0), 4)

            # Record data
            timestamps[t] = currentTimestamp()
            features[t] = len(kp)
            shutters[t] = cap.get(15)
            gains[t] = cap.get(14)

            cv2.imshow('frame', disp)

            t += 1

        if t >= totalFrames:
            break
        elif t < totalFrames - 1:
            cap.set(15, shutters[t+1])
            cap.set(14, gains[t+1])

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

# Put the data back into a dataframe
df = pd.DataFrame({ 'Timestamp': timestamps,
                    'Gain': gains,
                    'Shutter': shutters,
                    'ORB Features': features,
                    'Image File': imageFiles },
                    index=frames)

# Create the output directory and save the data
shortname = sys.argv[1][:-5]
for i in range(100):
    foldername = 'data/' + shortname + '-' + str(i)
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        break

df.to_csv(foldername + '/' + shortname + '_data.csv')
shutil.copy(sys.argv[1], foldername)
if 'settingsFile' in config:
    shutil.copy(config['settingsFile'], foldername)

