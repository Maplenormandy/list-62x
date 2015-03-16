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
import pickle
import imageFeatures as iF



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
	return grayImages,absentPictures,folder

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
	sift = cv2.SURF()
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary

	flann = cv2.FlannBasedMatcher(index_params,search_params)

	maxKeypoints = 0 
	baselinePosition = 0
	leftRight = 0  #left = 0, right = 1
	# for a in range(0, len(grayImgs)):

	for e in range(0, len(grayImgs[0])):
		keypoints, descriptors = sift.detectAndCompute(grayImgs[0][e], None)
		imgKeypoints.append(keypoints)
		imgDescriptors.append(descriptors)
		if (len(keypoints) > maxKeypoints):
			leftRight = 0
			baselinePosition = e
			maxKeypoints = len(keypoints)
	
	print "baseline picture is "+str(baselinePosition)+" on side "+str(leftRight)
	print "it has "+ str(maxKeypoints)+" features"

	return baselinePosition,leftRight,sift,flann


def determineBrightness(image):
	brightness=[]
	brightness.append(cv2.mean(image)[0])
	brightness.append(iF.contrast(image))
	# first index mean, second contrast
	return brightness

def findBestMatch(baselinePos, startVal, endVal, grayImages,sift,flann, leftRight):
	if endVal<startVal or endVal>len(grayImages[0]):
		print "Error - Incorrect function call"
		return
	keypointsBaseline,descriptorsBaseline = sift.detectAndCompute(grayImages[leftRight][baselinePos], None)
	baselineBrightness = determineBrightness(grayImages[leftRight][baselinePos])

	matchImageQualities=[]
	brightnessVals=[]
	maxPrecision=0
	bestImage = 0
	precisionValues=[]
	
	for c in range(startVal, endVal):
			keypoints,descriptors = sift.detectAndCompute(grayImages[1][c],None)
			if descriptors!= None and descriptorsBaseline!=None:
				matches = flann.knnMatch(descriptorsBaseline,descriptors,k=2)
				efficiency = usableMatch(matches, keypoints, keypointsBaseline)
				matchImageQualities.append(efficiency)
				brightnessVal = determineBrightness(grayImages[1][c])
				# print str(c)+ "length of "+str(brightnessVal)
				brightnessVals.append(brightnessVal)
			else:
				brightnessVals.append([0,0])
				matchImageQualities.append([0,0])
	
	side=1

	
	for d in range(0, len(matchImageQualities)):
		if (matchImageQualities[d][1]>50):
			precision  = float(matchImageQualities[d][0]/ matchImageQualities[d][1])
		else:
			precision = 0
		if precision > maxPrecision: #and (d+startVal)!=baselinePos:
			side = 1
			maxPrecision = precision
			bestImage = d+startVal
		precisionValues.append(precision)

	print str(bestImage)+" is best image on side "+str(side)+" with "+str(matchImageQualities[bestImage%50][0])+ " matches"
	bestBrightness = determineBrightness(grayImages[side][bestImage])
	allParamVals = [brightnessVals, matchImageQualities]
	return bestImage,side, bestBrightness,baselineBrightness,allParamVals



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


def getCameraSettingsData(fileName):
	df = pd.read_csv(fileName)
	return df

# ithm(grays)


def initializeDataFrame():
# create dataframe
	df = pd.DataFrame( columns=('Shutter 0', 'Gain 0', 'Mean Illumination 0','Contrast 0', 'Shutter 1', 'Gain 1', 'Mean Illumination 1', 'Contrast 1','FileLocation', 'NumMatches') )

	return df

def saveOptimalSettingsVector(df, baselinePos,sideBaseline, bestPos, sideBest, optimalPointsDf, bestBrightness,baselineBrightness, folder, allParamVals,i): #imageBrightness
	
	a=str(sideBaseline)
	b = str(sideBest)
	optimalPointsDf.loc[len(optimalPointsDf['Shutter 0'])] = ([df['Shutter '+a][baselinePos], df['Gain '+a][baselinePos], baselineBrightness[0], baselineBrightness[1], df['Shutter '+b][bestPos], df['Gain '+b][bestPos],bestBrightness[0], bestBrightness[1],folder,allParamVals[1][bestPos-i][0]])
	
	return optimalPointsDf


def initializeAllParamsDF():
	df = pd.DataFrame( columns=('Shutter B', 'Gain B', 'Mean Illumination B', 'Contrast B', 'Shutter I', 'Gain I', 'Mean Illumination I', 'Contrast I', 'numFeatures I', 'numMatches B-I', 'name I', 'FileLocation'))
	return df



def saveAllImageParameters(df, baselinePos, sideBaseline, allParamVals, dfParamVals,baselineBrightness, folder,x):
	a =str(sideBaseline)

	for j in range(0, len(allParamVals[0])):
		b = str(1)
		dfParamVals.loc[len(dfParamVals['Shutter B'])] = ([ df['Shutter '+a][baselinePos], df['Gain '+a][baselinePos], baselineBrightness[0], baselineBrightness[1], df['Shutter '+b][j], df['Gain '+b][j], allParamVals[0][j][0], allParamVals[0][j][1], allParamVals[1][j][1], allParamVals[1][j][0], df['Image File '+b][j+x], folder])

	return dfParamVals



def checkPictures(grays, missedPics):
	for i in range(0, len(missedPics)):
		for j in range(0, len(missedPics[i])):
			print "missed Picture at position "+ str(missedPics[i][j])
	return [grays, missedPics]


def iterateThruData(baselinePos,sideBaseline,iterator, grayImgs,sift,flann, optPointsdf,data, folder, dfParamVals):
	i=0
	while (i<=len(grayImgs[0])):
		j = i+iterator
		# print grayImgs
		bestPos,sideBest,bestBrightness,baselineBrightness, allParamVals = findBestMatch(baselinePos,i,j,grayImgs, sift,flann,sideBaseline)
		optPointsdf = saveOptimalSettingsVector(data, baselinePos,sideBaseline, bestPos,sideBest, optPointsdf,bestBrightness,baselineBrightness, folder, allParamVals,i)
		dfParamVals = saveAllImageParameters(data, baselinePos, sideBaseline, allParamVals, dfParamVals,baselineBrightness, folder,i)
		i=j+1
		
	return optPointsdf, dfParamVals
def addToCSV(fileName, optPointsData):
	if (os.path.exists(fileName)):
		print "adding to previous CSV"
		f = open(fileName, 'a') # Open file as append mode
		optPointsData.to_csv(f, header = False)
		f.close()
	else:
		print "creating new CSV"
		optPointsData.to_csv(fileName)

#using statsmodels.api library
def leastSquaresRegression(fileName, optParam,dependParam, Xparam):
	
	data = getCameraSettingsData(fileName)
	X = data[dependParam]
	Y = data[optParam] - data['Shutter 0']
	data.head()
	X = sm.add_constant(X)
	est = sm.OLS(Y,X).fit()
	print('Parameters: ', est.params)
	print('Standard errors: ', est.bse)
	print('Predicted values: ', est.predict())

	prstd, iv_l, iv_u = wls_prediction_std(est)

	fig, ax = plt.subplots(figsize=(8,6))

	ax.plot(X[Xparam] - X['Mean Illumination 0'], Y, 'o', label="Training Data")
	# ax.plot(X[Xparam]- X['Illumination 0'], est.fittedvalues, 'r--.', label="Least Squares")
	ax.legend(loc='best')
	plt.suptitle("Regression for Predicted Shutter Speed")
	plt.ylabel('Difference of Predicted Shutter Speed and Baseline Shutter Speed')
	plt.xlabel('Mean Brightness of Best Image - Mean Brightness of Baseline Image')
	plt.ylim([-600,600])
	plt.show()
 
	pickle.dump( est, open( "regression.p", "wb" ) )
	return est.summary()

def runTests(start, end):
	i=start
	while (i<=end):
		j = i+49
		grays0, missedPictures0, folder = prepData('csail03-08-15_0/','csail03-08-15_0_', i,j)
		grays1, missedPictures1, folder1 = prepData('csail03-08-15_0/','csail03-08-15_1_', i,j)
		picData = checkPictures([grays0,grays1], [missedPictures0, missedPictures1])
		baselinePos,sideBaseline, sift, flann = findBaseline(picData[0])
		data = getCameraSettingsData('csail03-08-15_0/csail03-08-15_rawdata.csv')
		optPointsdf = initializeDataFrame()
		dfParamVals = initializeAllParamsDF()
		optPointsdf, dfParamVals  = iterateThruData(baselinePos,sideBaseline, 49, picData[0],sift, flann,optPointsdf,data, folder, dfParamVals)
		addToCSV('optimalPoints.csv', optPointsdf)
		addToCSV('allParamterValues.csv', dfParamVals)
		i=j+1
	# print leastSquaresRegression('optimalPoints.csv','Shutter 1', ['Shutter 0','Gain 0', 'Mean Illumination 0', 'Mean Illumination 1'],'Mean Illumination 1' )



# runTests(0,499) #10
# print "finished 10"
# runTests(650,899) #5
# print "finished 15"
# runTests(950,1299) #7
# print finished "22"
# runTests(1450,1499) #1
# runTests(1550,1599) #1


# grays0, missedPictures0, folder = prepData('2015-02-22TNQ_1/','2015-02-22TNQ_0_', 0,99)
# grays1, missedPictures1, folder1 = prepData('2015-02-22TNQ_1/','2015-02-22TNQ_1_', 0,99)
# picData = checkPictures([grays0,grays1], [missedPictures0, missedPictures1])
# baselinePos,sideBaseline, sift, flann = findBaseline(picData[0])
# data = getCameraSettingsData('2015-02-22TNQ_1/2015-02-22TNQ_rawdata.csv')
# optPointsdf = initializeDataFrame()
# dfParamVals = initializeAllParamsDF()
# optPointsdf, dfParamVals  = iterateThruData(baselinePos,sideBaseline, 49, picData[0],sift, flann,optPointsdf,data, folder, dfParamVals)
# addToCSV('allParamterValues.csv', dfParamVals)
# addToCSV('optimalPoints.csv', optPointsdf)

# print leastSquaresRegression('optimalPoints.csv','Shutter 1', ['Shutter 0','Gain 0', 'Illumination 0', 'Illumination 1'],'Illumination 1' )


# grays0, missedPictures0, folder = prepData('2015-02-22TNQ_0/','2015-02-22TNQ_0_', 0,349)
# grays1, missedPictures1, folder1 = prepData('2015-02-22TNQ_0/','2015-02-22TNQ_1_', 0,349)
# picData = checkPictures([grays0,grays1], [missedPictures0, missedPictures1])
# baselinePos,sideBaseline, sift, flann = findBaseline(picData[0])
# data = getCameraSettingsData('2015-02-22TNQ_0/2015-02-22TNQ_rawdata.csv')
# optPointsdf = initializeDataFrame()
# dfParamVals = initializeAllParamsDF()
# optPointsdf,dfParamVals  = iterateThruData(baselinePos,sideBaseline, 49, picData[0],sift, flann,optPointsdf,data, folder,dfParamVals)

# addToCSV('allParamterValues.csv', dfParamVals)
# addToCSV('optimalPoints.csv', optPointsdf)

print leastSquaresRegression('optimalPoints.csv','Shutter 1', ['Shutter 0','Gain 0', 'Mean Illumination 0', 'Mean Illumination 1', 'Contrast 0', 'Contrast 1'],'Mean Illumination 1')

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


def runTests64(start, end):
	i=start
	while (i<=end):
		j = i+62
		grays0, missedPictures0, folder = prepData('csail03-08-15_0/','csail03-08-15_0_', i,j)
		grays1, missedPictures1, folder1 = prepData('csail03-08-15_0/','csail03-08-15_1_', i,j)
		picData = checkPictures([grays0,grays1], [missedPictures0, missedPictures1])
		baselinePos,sideBaseline, sift, flann = findBaseline(picData[0])
		data = getCameraSettingsData('csail03-08-15_0/csail03-08-15_rawdata.csv')
		optPointsdf = initializeDataFrame()
		dfParamVals = initializeAllParamsDF()
		optPointsdf, dfParamVals  = iterateThruData(baselinePos,sideBaseline, 62, picData[0],sift, flann,optPointsdf,data, folder, dfParamVals)
		addToCSV('optimalPoints.csv', optPointsdf)
		addToCSV('allParamterValues.csv', dfParamVals)
		i=j+2