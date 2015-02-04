# This metric is currently functional and works to determine the best pictures, but we 
# evaluate what our metric, currently it is usableMatches/foundKeyPoints for that image, 
# I had trouble doing it for total matches found ---.

import cv2
import numpy as np
from matplotlib import pyplot as plt

# array of all images except baseline
images = []
# images are now converted to grayscale for feature detection
grayImages=[]
matchImageQualities = []
bestImage = None
maxPrecision = 0
imgKeypoints=[]


def usableMatch(matches, keypoints):
	correctMatches = []
	for m,n in matches:
		if m.distance <.75*n.distance:
			correctMatches.append([m])
	efficiency = [correctMatches, len(keypoints)]
	return efficiency

# adding all color images to array
a=50
while 50<=a<=450:
	if a!=250:
		string = "image_"+str(a)+'.png'
		images.append(cv2.imread(string))
	a=a+50

# converting images to grayscale and adding to array
for b in range(0, len(images)):
	grayImg = cv2.cvtColor(images[b],cv2.COLOR_BGR2GRAY)
	grayImages.append(grayImg)


# arbitrarily setting baseline image
baseline = cv2.imread('image_250.png')
baselineGray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)

# using feature detector sift
sift = cv2.SIFT()
# feature detection on baseline image
keypointsBaseline,descriptorsBaseline = sift.detectAndCompute(baselineGray, None)
baseline = cv2.drawKeypoints(baselineGray,keypointsBaseline)

cv2.imwrite('sift_baseline.png', baseline)

# creating brute force feature matcher object
bf = cv2.BFMatcher()

# performing feature detection on all the other images and matching with baseline
for c in range(0, len(grayImages)):
	keypoints, descriptors = sift.detectAndCompute(grayImages[c], None)
	imgKeypoints.append(keypoints)
	matches = bf.knnMatch(descriptorsBaseline,descriptors, k=2)
	efficiency = usableMatch(matches, keypoints)
	matchImageQualities.append(efficiency)
	# print str(c)+ "length of "+str(len(descriptors))
	img = cv2.drawKeypoints(grayImages[c], keypoints)
	cv2.imwrite("image_"+str(c)+".png", img)
	

for d in range(0, len(matchImageQualities)):
	precision  = float(len(matchImageQualities[d][0]))/float(matchImageQualities[d][1])
	# print  str(d)+" and "+str(precision)
	if precision > maxPrecision:
		maxPrecision = precision
		bestImage = d

def findStringVal(bestImage):
	if (bestImage >=4):
		return (bestImage+2)*50
	else:
		return (bestImage+1)*50

print "best Image is "+str(findStringVal(bestImage))
# Have an error right here, this function, cv2.drawMatchesKnn, is not in older openCV versions

# bestImageNum = findStringVal(bestImage)
# bestImg = cv2.drawMatchesKnn(baseline,keypointsBaseline, images[bestImage],imgKeypoints[bestImage],matchImageQualities[d][1], flags=2)

# plt.imshow(bestImg), plt.show()


