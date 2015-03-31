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


def determineBrightness(image, kp):
	brightness=[]
	# struture of brightness - foreground, background, contrast, mean, featureLuminance, gaussian Blur
	brightness.append(iF.weightedLuminance(image))
	brightness.append(iF.contrast(image))
	brightness.append(iF.meanLuminance(image))
	brightness.append(iF.featureLuminance(image, kp))
	brightness.append(iF.gaussianBlurfeatureLuminance(image, kp))
	# first index mean, second contrast
	return brightness

def findBestMatch(baselinePos, startVal, endVal, grayImages,sift,flann, leftRight):
	if endVal<startVal or endVal>len(grayImages[0]):
		print "Error - Incorrect function call"
		return
	keypointsBaseline,descriptorsBaseline = sift.detectAndCompute(grayImages[leftRight][baselinePos], None)
	baselineBrightness = determineBrightness(grayImages[leftRight][baselinePos], keypointsBaseline)

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
				brightnessVal = determineBrightness(grayImages[1][c], keypoints)
				# print str(c)+ "length of "+str(brightnessVal)
				brightnessVals.append(brightnessVal)
			else:
				# brightnessVals.append([0,0])
				# struture of brightness - foreground, background, contrast, mean, featureLuminance
				brightnessVals.append([[0,0],0,0,0,0])
				matchImageQualities.append([0,0])
	
	side=1

	
	for d in range(0, len(matchImageQualities)):
		if (matchImageQualities[d][1]>50):
			precision  = float(matchImageQualities[d][0])
		else:
			precision = 0
		if precision > maxPrecision: #and (d+startVal)!=baselinePos:
			side = 1
			maxPrecision = precision
			bestImage = d+startVal
		precisionValues.append(precision)

	print str(bestImage)+" is best image on side "+str(side)+" with "+str(matchImageQualities[bestImage%50][0])+ " matches"
	keypointsBest,descriptorsBest = sift.detectAndCompute(grayImages[side][bestImage], None)
	bestBrightness = determineBrightness(grayImages[side][bestImage], keypointsBest)
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
	# df = pd.DataFrame( columns=('Shutter 0', 'Gain 0', 'Mean Illumination 0','Contrast 0', 'Shutter 1', 'Gain 1', 'Mean Illumination 1', 'Contrast 1','FileLocation', 'NumMatches') )
	df = pd.DataFrame( columns=('Shutter 0', 'Gain 0', 'Mean Foreground Illumination 0', 'Mean BackGround Illumination 0','Contrast 0', 'Mean 0', 'Feature Luminance 0','Blur Luminance 0', 'Shutter 1', 'Gain 1', 'Mean Foreground Illumination 1', 'Mean BackGround Illumination 1', 'Contrast 1','Mean 1','Feature Luminance 1','Blur Luminance 1', 'FileLocation', 'NumMatches') )

	return df

def saveOptimalSettingsVector(df, baselinePos,sideBaseline, bestPos, sideBest, optimalPointsDf, bestBrightness,baselineBrightness, folder, allParamVals,i): #imageBrightness
	
	a=str(sideBaseline)
	b = str(sideBest)
	# optimalPointsDf.loc[len(optimalPointsDf['Shutter 0'])] = ([df['Shutter '+a][baselinePos], df['Gain '+a][baselinePos], baselineBrightness[0], baselineBrightness[1], df['Shutter '+b][bestPos], df['Gain '+b][bestPos],bestBrightness[0], bestBrightness[1],folder,allParamVals[1][bestPos-i][0]])
	optimalPointsDf.loc[len(optimalPointsDf['Shutter 0'])] = ([df['Shutter '+a][baselinePos], df['Gain '+a][baselinePos], baselineBrightness[0][0],baselineBrightness[0][1], baselineBrightness[1],baselineBrightness[2], baselineBrightness[3],baselineBrightness[4], df['Shutter '+b][bestPos], df['Gain '+b][bestPos],bestBrightness[0][0], bestBrightness[0][1], bestBrightness[1],bestBrightness[2], bestBrightness[3],bestBrightness[4], folder,allParamVals[1][bestPos-i][0]])

	return optimalPointsDf


def initializeAllParamsDF():
	df = pd.DataFrame( columns=('Shutter B', 'Gain B', 'Mean ForeGround Illumination B', 'Mean Background Illumination B', 'Contrast B','Mean B', 'Feature Luminance B','Blur Luminance B', 'Shutter I', 'Gain I', 'Mean Foreground Illumination I','Mean Background Illumination I', 'Contrast I','Mean I', 'Feature Luminance I','Blur Luminance I', 'numFeatures I', 'numMatches B-I', 'name I', 'FileLocation'))
	return df



def saveAllImageParameters(df, baselinePos, sideBaseline, allParamVals, dfParamVals,baselineBrightness, folder,x):
	a =str(sideBaseline)

	for j in range(0, len(allParamVals[0])):
		b = str(1)
		# dfParamVals.loc[len(dfParamVals['Shutter B'])] = ([ df['Shutter '+a][baselinePos], df['Gain '+a][baselinePos], baselineBrightness[0], baselineBrightness[1], df['Shutter '+b][j], df['Gain '+b][j], allParamVals[0][j][0], allParamVals[0][j][1], allParamVals[1][j][1], allParamVals[1][j][0], df['Image File '+b][j+x], folder])
		# print "error checking"
		# print baselineBrightness[0][0]
		# print baselineBrightness[0][1]
		# print allParamVals[0][j][0][0]
		# print allParamVals[0][j][0][1]
		dfParamVals.loc[len(dfParamVals['Shutter B'])] = ([ df['Shutter '+a][baselinePos], df['Gain '+a][baselinePos], baselineBrightness[0][0], baselineBrightness[0][1], baselineBrightness[1], baselineBrightness[2], baselineBrightness[3], baselineBrightness[4], df['Shutter '+b][j], df['Gain '+b][j], allParamVals[0][j][0][0], allParamVals[0][j][0][1], allParamVals[0][j][1], allParamVals[0][j][2], allParamVals[0][j][3], allParamVals[0][j][4], allParamVals[1][j][1], allParamVals[1][j][0], df['Image File '+b][j+x], folder])

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

def addExposure(filename):
	data = getCameraSettingsData(filename)
	data['Exposure 0'] = data['Shutter 0']* data['Gain 0']/16.0
	data['Exposure 1'] = data['Shutter 1']*data['Gain 1']/16.0
	return data

#using statsmodels.api library
# def leastSquaresRegression(fileName, optParam,independParam, Xparam, data):
	
# 	X = data[independParam]
# 	Y = data[optParam] #- data['Shutter 0']
# 	data.head()
# 	X = sm.add_constant(X)
# 	est = sm.OLS(Y,X).fit()
# 	print('Parameters: ', est.params)
# 	print('Standard errors: ', est.bse)
# 	print('Predicted values: ', est.predict())

# 	prstd, iv_l, iv_u = wls_prediction_std(est)

# 	fig, ax = plt.subplots(figsize=(8,6))

# 	ax.plot(X[Xparam], Y, 'o', label="Training Data")
# 	# ax.plot(X[Xparam]- X['Illumination 0'], est.fittedvalues, 'r--.', label="Least Squares")
# 	ax.legend(loc='best')
# 	plt.suptitle("Regression for Predicted Shutter Speed")
# 	plt.ylabel('Predicted Shutter Speed ')
# 	plt.xlabel('Mean Illumination of Best Image')
# 	plt.ylim([0,600])
# 	plt.show()
 
# 	pickle.dump( est, open( "regression.p", "wb" ) )
# 	return est.summary()
def differencesLeastSquaresRegression(fileName, optParam,independParam, Xparam, data):
	
	X = data[independParam]
	Y = data[optParam] #- data['Shutter 0']
	data.head()
	X = sm.add_constant(X)
	est = sm.OLS(Y,X).fit()
	# print('Parameters: ', est.params)
	# print('Standard errors: ', est.bse)
	# print('Predicted values: ', est.predict())

	# prstd, iv_l, iv_u = wls_prediction_std(est)

	# fig, ax = plt.subplots(figsize=(8,6))

	# ax.plot(X[Xparam]-X['Feature Luminance 0'], Y, 'o', label="Training Data")
	# # ax.plot(X[Xparam]- X['Illumination 0'], est.fittedvalues, 'r--.', label="Least Squares")
	# ax.legend(loc='best')
	# plt.suptitle("Regression for Predicted Shutter Speed")
	# plt.ylabel('Predicted Shutter Speed ')
	# plt.xlabel('Mean Brightness of Best Image')
	# plt.ylim([0,600])
	# plt.show()
 
	pickle.dump( est, open( "mean_foreground_luminance.p", "wb" ) )
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
		addToCSV('optimalPointsWmask.csv', optPointsdf)
		addToCSV('allParamterValuesWmask.csv', dfParamVals)
		i=j+1
	# print leastSquaresRegression('optimalPoints.csv','Shutter 1', ['Shutter 0','Gain 0', 'Mean Illumination 0', 'Mean Illumination 1'],'Mean Illumination 1' )



# runTests(0,499) #10
# print "finished 10"
# runTests(650,899) #5
# print "finished 15"
# runTests(950,1299) #7
# print "finished 22"
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
# addToCSV('allParamterValuesWmask.csv', dfParamVals)
# addToCSV('optimalPointsWmask.csv', optPointsdf)



# grays0, missedPictures0, folder = prepData('2015-02-22TNQ_0/','2015-02-22TNQ_0_', 0,349)
# grays1, missedPictures1, folder1 = prepData('2015-02-22TNQ_0/','2015-02-22TNQ_1_', 0,349)
# picData = checkPictures([grays0,grays1], [missedPictures0, missedPictures1])
# baselinePos,sideBaseline, sift, flann = findBaseline(picData[0])
# data = getCameraSettingsData('2015-02-22TNQ_0/2015-02-22TNQ_rawdata.csv')
# optPointsdf = initializeDataFrame()
# dfParamVals = initializeAllParamsDF()
# optPointsdf,dfParamVals  = iterateThruData(baselinePos,sideBaseline, 49, picData[0],sift, flann,optPointsdf,data, folder,dfParamVals)

# addToCSV('allParamterValuesWmask.csv', dfParamVals)
# addToCSV('optimalPointsWmask.csv', optPointsdf)


def runTests64(start, end):
	i=start
	while (i<=end):
		j = i+62
		grays0, missedPictures0, folder = prepData('2015-03-14gelb_2/','2015-03-14gelb_0_', i,j)
		grays1, missedPictures1, folder1 = prepData('2015-03-14gelb_2/','2015-03-14gelb_1_', i,j)
		picData = checkPictures([grays0,grays1], [missedPictures0, missedPictures1])
		baselinePos,sideBaseline, sift, flann = findBaseline(picData[0])
		data = getCameraSettingsData('2015-03-14gelb_2/2015-03-14gelb_rawdata.csv')
		optPointsdf = initializeDataFrame()
		dfParamVals = initializeAllParamsDF()
		optPointsdf, dfParamVals  = iterateThruData(baselinePos,sideBaseline, 62, picData[0],sift, flann,optPointsdf,data, folder, dfParamVals)
		addToCSV('optimalPointsWmask.csv', optPointsdf)
		addToCSV('allParamterValuesWmask.csv', dfParamVals)
		i=j+2
# skip 0,128,512
# runTests64(64,127)
# runTests64(192,511)
# runTests64(576,1279)

data = addExposure('optimalPointsWmask.csv')
# print differencesLeastSquaresRegression('optimalPointsWmask.csv','Exposure 1', ['Exposure 0', 'Mean 0', 'Mean 1'],'Mean 1', data)
# print differencesLeastSquaresRegression('optimalPointsWmask.csv','Exposure 1', ['Exposure 0', 'Mean Foreground Illumination 0','Mean BackGround Illumination 0','Mean BackGround Illumination 1', 'Mean Foreground Illumination 1'],'Mean Foreground Illumination 1', data)
print differencesLeastSquaresRegression('optimalPointsWmask.csv','Exposure 1', ['Exposure 0','Mean Foreground Illumination 0','Mean Foreground Illumination 1'], 'Mean Foreground Illumination 1', data)


