# This metric is currently functional  works to determine the best pictures, but we 
# evaluate what our metric, currently it is usableMatches/foundKeyPoints for that image, 
# I had trouble doing it for total matches found ---.

import cv2
import os.path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn import datasets, linear_model



# '2015-02-22TNQ_1/2015-02-22TNQ_0_0000.png
global images
global grayImages

def prepData(folder, pictureStartString, start, end):
	images = []
	grayImages=[]
	absentPictures =[]
	for a in range(start,end):
		picString = folder+pictureStartString+formatNumber(a)+".png"
		if (os.path.exists(picString)):
			images.append(cv2.imread(picString))
			grayImages.append(cv2.cvtColor(cv2.imread(picString), cv2.COLOR_BGR2GRAY))
		else:
			absentPictures.append(a)
	return grayImages,absentPictures

def formatNumber(x):
	if (x==0):
		x ="0000"
	elif (x<10):
		x= "000"+str(x)
	elif(x>=10 and x<100):
		x= "00"+str(x)
	elif(x>=100 and x<1000):
		x = "0"+str(x)
	else:
		x = str(x)
	return x

# adding all color images to array
# array of all images except baseline

def findBaseline(grayImgs):
	imgKeypoints = []
	imgDescriptors=[]
	precisionValues=[]
	# converting images to grayscale and adding to array

	# using feature detector orb -  could change to sift or surf if necessary.
	sift = cv2.SURF()
	# feature detection on baseline image
	# # creating brute force feature matcher object
	#  = cv2.BFMatcher()
	# FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary

	flann = cv2.FlannBasedMatcher(index_params,search_params)

	# performing feature detection on all the other images and matching with baseline
	maxKeypoints = 0 
	baselinePosition = 0
	leftRight = 0  #left = 0, right = 1
	for a in range(0, len(grayImgs)):
		keypts=[]
		dscriptors=[]
		for e in range(0, len(grayImgs[a])):
			keypoints, descriptors = sift.detectAndCompute(grayImgs[a][e], None)
			keypts.append(keypoints)
			dscriptors.append(descriptors)
			if (len(keypoints) > maxKeypoints):
				leftRight = a
				baselinePosition = e
				maxKeypoints = len(keypoints)
		imgKeypoints.append(keypts)
		imgDescriptors.append(dscriptors)
	print "baseline picture is "+str(baselinePosition)
	print "it has "+ str(maxKeypoints)+" features"

	return baselinePosition,leftRight,sift,flann

def findBestMatch(baselinePos, startVal, endVal, grayImages,sift,flann, leftRight):
	if endVal<startVal or endVal>len(grayImages[0]):
		print "Error - Incorrect function call"
		return
	keypointsBaseline,descriptorsBaseline = sift.detectAndCompute(grayImages[leftRight][baselinePos], None)
	baselineBrightness = cv2.mean(grayImages[leftRight][baselinePos])[0]

	matchImageQualities=[]
	imgBrightness=[]
	maxPrecision=0
	bestImage = 0
	precisionValues=[]
	for b in range(0, len(grayImages)):
		matchQuals=[]
		brightList=[]
		for c in range(startVal, endVal):
				keypoints,descriptors = sift.detectAndCompute(grayImages[b][c],None)
				if descriptors!= None and descriptorsBaseline!=None:
					matches = flann.knnMatch(descriptorsBaseline,descriptors,k=2)
					efficiency = usableMatch(matches, keypoints, keypointsBaseline)
					matchQuals.append(efficiency)
					brightnessVal = cv2.mean(grayImages[b][c])[0]
					# print str(c)+ "length of "+str(brightnessVal)
					brightList.append(brightnessVal)
				else:
					brightList.append(0)
					matchQuals.append([0,0])
		matchImageQualities.append(matchQuals)
		imgBrightness.append(brightList)
	side=0
	for e in range(0, len(matchImageQualities)):
		prec=[]
		for d in range(0, len(matchImageQualities[e])):
			if (matchImageQualities[e][d][1]>1):
				precision  = float(matchImageQualities[e][d][0])/float(matchImageQualities[e][d][1])
			else:
				precision = 0
			prec.append(precision)
			if precision > maxPrecision and (d+startVal)!=baselinePos:
				side = e
				maxPrecision = precision
				bestImage = d+startVal
		precisionValues.append(prec)
	print str(bestImage)+" is best image on side "+str(side)
	bestBrightness = cv2.mean(grayImages[side][bestImage])[0]
	return bestImage,side, bestBrightness,baselineBrightness



def usableMatch(matches, keypoints, keypointsBaseline):
	correctMatches = []
	minAmmount = 5
	srcPts=[]
	dstPts=[]
	for m,n in matches:
		if m.distance <.75*n.distance:
			correctMatches.append(m)
	# for m in correctMatches:
	# 	print "entered"
	# 	print 
	# 	dstPts.append(np.float32([keypoints[m.trainIdx].pt]))
	# 	srcPts.append(np.float32([keypointsBaseline[m.queryIdx].pt]))

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

# def filter_matches(kp1, kp2, matches, ratio = 0.75):
#     """
#     Keep only matches that have distance ratio to 
#     second closest point less than 'ratio'.
#     """
#     mkp1, mkp2 = [], []
#     for m in matches:
#         if m[0].distance < m[1].distance * ratio:
#             m = m[0]
#             mkp1.append( kp1[m.queryIdx] )
#             mkp2.append( kp2[m.trainIdx] )
#     p1 = np.float32([kp.pt for kp in mkp1])
#     p2 = np.float32([kp.pt for kp in mkp2])
#     kp_pairs = zip(mkp1, mkp2)
#     return p1, p2, kp_pairs

#using statsmodels.api library
def leastSquaresRegression(fileName, optParam,dependParam, Xparam):
	
	data = getCameraSettingsData(fileName)
	X = data[dependParam]
	Y = data[optParam]
	data.head()
	X = sm.add_constant(X)
	est = sm.OLS(Y,X).fit()
	print('Parameters: ', est.params)
	print('Standard errors: ', est.bse)
	print('Predicted values: ', est.predict())

	prstd, iv_l, iv_u = wls_prediction_std(est)

	fig, ax = plt.subplots(figsize=(8,6))

	ax.plot(X[Xparam], Y, 'o', label="Training Data")
	ax.plot(X[Xparam], est.fittedvalues, 'r--.', label="Least Squares")
	ax.legend(loc='best')
	plt.suptitle("Regression for Predicted Shutter Speed")
	plt.ylabel('Predicted Shutter Speed')
	plt.xlabel('Mean Brightness of Best Image')
	plt.show()
	return est.summary()
def getCameraSettingsData(fileName):
	df = pd.read_csv(fileName)
	return df

# ithm(grays)
def initializeDataFrame():
	
# create dataframe
	df = pd.DataFrame( columns=('Shutter 0', 'Gain 0', 'Illumination 0', 'Shutter 1', 'Gain 1', 'Illumination 1') )

	return df


def saveOptimalSettingsVector(df, baselinePos,sideBaseline, bestPos, sideBest, optimalPointsDf, bestBrightness,baselineBrightness): #imageBrightness
	# Feature Vector = [[baseIllumination, bestIllumination, baseShutter, baseGain, baseVariac, bestVariac],[bestShutter, bestGain]]
	# baseIllumination = 
	a=str(sideBaseline)
	b = str(sideBest)
	optimalPointsDf.loc[len(optimalPointsDf['Shutter 0'])] = ([df['Shutter '+a][baselinePos],df['Gain '+a][baselinePos],baselineBrightness,df['Shutter '+b][bestPos], df['Gain '+b][bestPos],bestBrightness])
	
	return optimalPointsDf

def checkPictures(grays, missedPics):
	for i in range(0, len(missedPics)):
		for j in range(0, len(missedPics[i])):
			print "missed Picture at position "+ str(missedPictures[i][j])
	return [grays, missedPics]


def iterateThruData(baselinePos,sideBaseline,iterator, grayImgs,sift,flann, optPointsdf,data):
	i=0
	while (i<=len(grayImgs[0])):
		j = i+iterator
		
		# print grayImgs
		bestPos,sideBest,bestBrightness,baselineBrightness = findBestMatch(baselinePos,i,j,grayImgs, sift,flann,sideBaseline)
		optPointsdf = saveOptimalSettingsVector(data, baselinePos,sideBaseline, bestPos,sideBest, optPointsdf,bestBrightness,baselineBrightness)
		i=j+1
	return optPointsdf
def addToCSV(fileName, optPointsData):
	if (os.path.exists(fileName)):
		print "adding to previous CSV"
		f = open(fileName, 'a') # Open file as append mode
		optPointsData.to_csv(f, header = False)
		f.close()
	else:
		print "creating new CSV"
		optPointsData.to_csv(fileName)

grays0, missedPictures0 = prepData('2015-02-22TNQ_0/','2015-02-22TNQ_0_', 0,349)
grays1, missedPictures1 = prepData('2015-02-22TNQ_0/','2015-02-22TNQ_1_', 0,349)
picData = checkPictures([grays0,grays1], [missedPictures0, missedPictures1])
baselinePos,sideBaseline, sift, flann = findBaseline(picData[0])
data = getCameraSettingsData('2015-02-22TNQ_0/2015-02-22TNQ_rawdata.csv')
optPointsdf = initializeDataFrame()

optPointsdf  = iterateThruData(baselinePos,sideBaseline, 49, picData[0],sift, flann,optPointsdf,data)
addToCSV('optimalPoints.csv', optPointsdf)
# optPointsdf.to_csv('optimalPoints.csv')

print leastSquaresRegression('optimalPoints.csv','Shutter 1', ['Shutter 0','Gain 0', 'Illumination 0', 'Illumination 1'],'Illumination 1' )


# imgX = cv2.imread('2015-02-22TNQ_0/2015-02-22TNQ_0_0011.png')
# img1 = cv2.imread('2015-02-22TNQ_0/2015-02-22TNQ_0_0016.png')

# imgGray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# imgGray2 = cv2.cvtColor(imgX, cv2.COLOR_BGR2GRAY)
# sift = cv2.SURF()
# # feature detection on baseline image
# # # creating brute force feature matcher object
# #  = cv2.BFMatcher()
# # FLANN parameters
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary

# flann = cv2.FlannBasedMatcher(index_params,search_params)

# keypoints1,descriptors1 = sift.detectAndCompute(imgGray,None)
# keypoints2,descriptors2 = sift.detectAndCompute(imgGray2,None)
# matches = flann.knnMatch(descriptors1,descriptors2,k=2)
# # print matches[10][0].distance
# # print matches[10][0].trainIdx
# # print matches[10][0].queryIdx
# # print matches[10][0].imgIdx 
# # print matches[10][1].trainIdx
# # print matches[10][1].queryIdx
# print np.float32([keypoints1[matches[0][0].queryIdx].pt])
