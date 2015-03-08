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
		
			# print picString
			# print images
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

def runAlgorithm(grayImgs):
	imgKeypoints = []
	imgDescriptors=[]
	precisionValues=[]
	# converting images to grayscale and adding to array

	# using feature detector orb -  could change to sift or surf if necessary.
	sift = cv2.SURF()
	# feature detection on baseline image
	# creating brute force feature matcher object
	bf = cv2.BFMatcher()
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

	return baselinePosition,leftRight,sift,bf

def findBestMatch(baselinePos, startVal, endVal, grayImages,sift,bf, leftRight):
	if endVal<startVal or endVal>len(grayImages[0]):
		print "Error - Incorrect function call"
		return
	keypointsBaseline,descriptorsBaseline = sift.detectAndCompute(grayImages[leftRight][baselinePos], None)
	baselineBrightness = cv2.adaptiveThreshold(grayImages[leftRight][baselinePos],255,cv2.ADAPTIVE_THRESH_MEAN_C,\
	            cv2.THRESH_BINARY,11,2).mean() 

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
					matches = bf.knnMatch(descriptorsBaseline,descriptors, k=2)
					efficiency = usableMatch(matches, keypoints)
					matchQuals.append(efficiency)
					brightnessVal = cv2.adaptiveThreshold(grayImages[b][c],255,cv2.ADAPTIVE_THRESH_MEAN_C,\
				            cv2.THRESH_BINARY,11,2).mean()
					# print str(c)+ "length of "+str(brightnessVal)
					brightList.append(brightnessVal)
				else:
					brightList.append(0)
					matchQuals.append([[],0])
		matchImageQualities.append(matchQuals)
		imgBrightness.append(brightList)
	#print matchImageQualities
	side=0
	for e in range(0, len(matchImageQualities)):
		prec=[]
		for d in range(0, len(matchImageQualities[e])):
			if (matchImageQualities[e][d][1]>1):
				precision  = float(len(matchImageQualities[e][d][0]))/float(matchImageQualities[e][d][1])
			else:
				precision = 0
			prec.append(precision)
			if precision > maxPrecision and (d+startVal)!=baselinePos:
				side = e
				maxPrecision = precision
				bestImage = d+startVal
		precisionValues.append(prec)
	print str(bestImage)+" is best image on side "+str(side)
	bestBrightness = baselineBrightness = cv2.adaptiveThreshold(grayImages[side][bestImage],255,cv2.ADAPTIVE_THRESH_MEAN_C,\
	            cv2.THRESH_BINARY,11,2).mean() 
	return bestImage,side, bestBrightness,baselineBrightness


def usableMatch(matches, keypoints):
	correctMatches = []
	for m,n in matches:
		if m.distance <.75*n.distance:
			correctMatches.append([m])
	efficiency = [correctMatches, len(keypoints)]
	return efficiency

#using sci-kit learn library
def leastSquaresRegression(dataX, dataY):
	# # Create linear regression object
	# regression = linear_model.LinearRegression()

	# # Train the model using the training sets
	# regression.fit(dataX, dataY)

	# # The coefficients
	# print('Coefficients: \n', regression.coef_)

	# # Plot outputs
	# plt.scatter(dataX, dataY,  color='black')
	# plt.plot(dataX, regr.predict(dataX), color='blue',linewidth=3)

	# plt.xticks(())
	# plt.yticks(())

	# plt.show()
	data = getCameraSettingsData('optimalPoints.csv')
	X = data[['Shutter 0','Gain 0', 'Illumination 0', 'Illumination 1']]
	Y = data['Shutter 1']
	data.head()
	X = sm.add_constant(X)
	est = sm.OLS(Y,X).fit()
	print('Parameters: ', est.params)
	print('Standard errors: ', est.bse)
	print('Predicted values: ', est.predict())

	prstd, iv_l, iv_u = wls_prediction_std(est)

	fig, ax = plt.subplots(figsize=(8,6))

	ax.plot(X['Illumination 1'], Y, 'o', label="Training Data")
	ax.plot(X['Illumination 1'], est.fittedvalues, 'r--.', label="Least Squares")
	ax.legend(loc='best')
	plt.suptitle("Regression for Predicted Shutter Speed")
	plt.ylabel('Difference of Shutter Speed')
	plt.xlabel('Difference of Illumination Value')
	plt.show()
	return est.summary()
def getCameraSettingsData(fileName):
	locString = fileName
	df = pd.read_csv(locString)
	return df

# runAlgorithm(grays)
def initializeDataFrame():
	
# create dataframe
	df = pd.DataFrame( columns=('Shutter 0', 'Gain 0', 'Illumination 0', 'Shutter 1', 'Gain 1', 'Illumination 1') )

	print df
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


def iterateThruData(baselinePos,sideBaseline,iterator, grayImgs,sift,bf, optPointsdf,data):
	i=0
	while (i<=len(grayImgs[0])):
		j = i+iterator
		
		# print grayImgs
		bestPos,sideBest,bestBrightness,baselineBrightness = findBestMatch(baselinePos,i,j,grayImgs, sift,bf,sideBaseline)
		optPointsdf = saveOptimalSettingsVector(data, baselinePos,sideBaseline, bestPos,sideBest, optPointsdf,bestBrightness,baselineBrightness)
		i=j+1
	return optPointsdf

# grays0, missedPictures0 = prepData('2015-02-22TNQ_0/','2015-02-22TNQ_0_', 0,349)
# grays1, missedPictures1 = prepData('2015-02-22TNQ_0/','2015-02-22TNQ_1_', 0,349)
# picData = checkPictures([grays0,grays1], [missedPictures0, missedPictures1])
# baselinePos,sideBaseline, sift, bf = runAlgorithm(picData[0])
# data = getCameraSettingsData('2015-02-22TNQ_0/','2015-02-22TNQ_rawdata.csv')
# optPointsdf = initializeDataFrame()

# optPointsdf  = iterateThruData(baselinePos,sideBaseline, 49, picData[0],sift, bf,optPointsdf,data)
# optPointsdf.to_csv('optimalPoints.csv')

print leastSquaresRegression('a','b')
