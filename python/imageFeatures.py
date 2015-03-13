import cv2
import np

def meanLuminance(grayImg):
    return cv2.mean(grayImg)[0]

def contrast(grayImg):
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grayImg)
    return np.Array([minVal, maxVal])
