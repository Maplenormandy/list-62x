import pandas as pd
import matplotlib.pyplot as plt


from statsmodels.stats.weightstats import ttost_paired
import statsmodels.api as sm
import numpy as np
from sklearn import datasets, linear_model
from pylab import *

data = pd.read_csv(open('optimalPointsWmask_16s.csv'), index_col='Num')
# print data['Mean 0']

#Baseline Mean Illumination vs Shutter
plt.scatter(data['Mean 0'], data['Shutter 0'], label='baseline')
x = data['Mean 0']
y = data['Shutter 0']
plt.legend(loc='upper right')
plt.ylabel('Shutter Speed')
plt.xlabel('Mean Illumination')
plt.title('Baseline Illumination vs Shutter')
(m, b) = polyfit(x,y,1)
yp= polyval([m,b], x)
plt.plot(x, yp, color='red')
plt.draw()
plt.show()

# Training Function Inputs --(Best Images) - Mean Illumination vs Shutter
plt.scatter(data['Mean 1'], data['Shutter 1'], label='training')
x = data['Mean 1']
y = data['Shutter 1']
plt.legend(loc='upper right')
plt.ylabel('Shutter Speed')
plt.xlabel('Mean Illumination')
plt.title('Training Illumination vs Shutter')
(m, b) = polyfit(x,y,1)
yp= polyval([m,b], x)
plt.plot(x, yp, color='red')
plt.draw()
plt.show()

#Baseline Mean Illumination vs Gain
plt.scatter(data['Mean 0'], data['Gain 0'], label='baseline')
x = data['Mean 0']
y= data['Gain 0']
plt.legend(loc='upper right')
plt.ylabel('Gain')
plt.xlabel('Mean Illumination')
plt.title('Baseline Illumination vs Gain')
(m, b) = polyfit(x,y,1)
yp= polyval([m,b], x)
plt.plot(x, yp, color='red')
plt.draw()
plt.show()

# Training Function Inputs --(Best Images) - Mean Illumination vs Shutter
plt.scatter(data['Mean 1'], data['Gain 1'], label='training')
x = data['Mean 1']
y = data['Gain 1']
plt.legend(loc='upper right')
plt.ylabel('Gain')
plt.xlabel('Mean Illumination')
plt.title('Training Illumination vs Gain')
(m, b) = polyfit(x,y,1)
yp= polyval([m,b], x)
plt.plot(x, yp, color='red')
plt.draw()
plt.show()
