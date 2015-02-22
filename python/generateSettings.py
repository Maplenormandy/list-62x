import cv2
import pandas as pd
import numpy as np

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cv2.namedWindow('frame')

if cap0.isOpened() and cap1.isOpened():

