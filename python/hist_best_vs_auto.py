import pandas as pd
import matplotlib.pyplot as plt


from statsmodels.stats.weightstats import ttost_paired
import statsmodels.api as sm
import numpy as np
from sklearn import datasets, linear_model
from pylab import *

dataAuto = pd.read_csv(open('autogain_points_2.csv'), index_col='Num')
dataOptimal = pd.read_csv(open('optimalPointsWmask_16s_testing_2.csv'), index_col='Num')
dataOptimalAll = pd.read_csv(open('optimalPointsWmask_16s.csv'), index_col='Num')
# print dataOptimal['NumMatches']
# print dataAuto['numMatches B-I']
plt.figure()
plt.hist(dataOptimal['NumMatches'] - dataAuto['numMatches B-I'], alpha = 0.5, bins=20, label="optimal training exposure-baseline", color="red")
plt.legend(loc='upper right')
plt.ylim(0,5)
plt.xlabel('Differential of Number of Matches')
plt.title('Histogram of Differential of Number of Matches')
plt.show()

plt.figure
plt.scatter(dataOptimalAll['Mean 1'], dataOptimalAll['Shutter 1']*dataOptimalAll['Gain 1']/16.0, label='training')
plt.legend(loc='upper right')
plt.ylabel('Exposure')
plt.xlabel('Mean Illumination')
plt.title('Mean Illumination vs Best Training Exposure')
# (m, b) = polyfit(x,y,1)
# print (m,b)
# yp= polyval([m,b], x)
# plt.plot(x, yp, color='red')
plt.draw()
plt.show()