import cv2
import math
import pandas as pd
import numpy as np
import time, sys, os, shutil
import yaml
from multiprocessing import Process, Queue
from Queue import Empty

# Change 0 to the index that works
cap = cv2.VideoCapture(0)

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if ret:
            disp = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
            cv2.imshow('frame', disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            # Turn on auto exposure
            cap.set(14, -2)
            cap.set(15, -2)
        elif key == ord('e'):
            # Turn off auto exposure
            cap.set(14, 16)
            cap.set(15, 1)
        elif key == ord('r'):
            # Turn off white balance
            cap.set(17, -4)
            cap.set(26, -4)
