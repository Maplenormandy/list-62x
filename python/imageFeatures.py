import cv2
import numpy as np

def meanLuminance(grayImg, kp=None):
    return cv2.mean(grayImg)[0]

def contrast(grayImg, kp=None):
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(grayImg)
    # return np.Array([minVal, maxVal])
    return (maxVal - minVal)

def weightedLuminance(grayImg, kp=None):
    """
    Foreground is middle 1/2x1/2 and bottom 1/4 strip
    Background is the rest
    """

    wlMaskFg = np.zeros(grayImg.shape, dtype=grayImg.dtype)
    height, width = grayImg.shape
    cv2.rectangle(wlMaskFg, (width/4, height/4), (3*width/4, 3*height/4), 255, -1)
    cv2.rectangle(wlMaskFg, (0, 3*height/4), (width, 3*height/4), 255, -1)
    wlMaskBg = cv2.bitwise_not(wlMaskFg)
    meanFg = cv2.mean(grayImg, mask=wlMaskFg)
    meanBg = cv2.mean(grayImg, mask=wlMaskBg)
    return (meanFg[0], meanBg[0])


def featureLuminance(grayImg, kp):
    avg = 0
    for i in range(len(kp)):
        x = int(kp[i].pt[0]+.5)
        y = int(kp[i].pt[1]+.5)
        avg += grayImg[y,x]*1.0
    return avg/len(kp)
def gaussianBlurfeatureLuminance(grayimg, kp):
	avg = 0
	blurImage = cv2.GuassianBlur(grayImg, 0, 5)
    for i in range(len(kp)):
        x = int(kp[i].pt[0]+.5)
        y = int(kp[i].pt[1]+.5)
        avg +=blurImage[y,x]*1.0
    return avg/len(kp)


def settingsToExposure(shutter, gain):
    return shutter*gain/16.0

def exposureToSettings(exposure):
    """
    Returns shutter, gain in a tuple
    """

    shutter = min(exposure, 531.0)
    gain = exposure / shutter * 16.0
    return shutter, gain
