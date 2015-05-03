import numpy as np
import pandas as pd
from sklearn import gaussian_process

import pickle

import matplotlib.pyplot as plt

data = pd.read_csv('optimalPointsWmask_16s.csv')
data['Exposure 0'] = data['Shutter 0']*data['Gain 0']/16.0
data['Exposure 1'] = data['Shutter 1']*data['Gain 1']/16.0

#params = ['Exposure 0', 'Mean 0', 'Mean 1']
#params = ['Exposure 0', 'Mean 0', 'Mean 1', 'Contrast 0', 'Contrast 1', 'Blur Luminance 0', 'Blur Luminance 1', 'Mean Foreground Illumination 0', 'Mean BackGround Illumination 0', 'Mean Foreground Illumination 1', 'Mean BackGround Illumination 1']
params = ['Exposure 0', 'Contrast 0', 'Contrast 1', 'Blur Luminance 0', 'Blur Luminance 1', 'Mean Foreground Illumination 0', 'Mean BackGround Illumination 0', 'Mean Foreground Illumination 1', 'Mean BackGround Illumination 1']

resids = [None] * 10

X = data[params].values
y = np.array(data['Exposure 1'])

gp = gaussian_process.GaussianProcess(nugget=10.0)
gp.fit(X,y)

# Use bootstrapping to get a more full error distribution
for i in range(10):
    data['rng'] = pd.Series(np.random.rand(len(data)), index=data.index)
    sampledData = data[data['rng']<0.9]

    X = sampledData[params].values
    y = np.array(sampledData['Exposure 1'])
    gp_test = gaussian_process.GaussianProcess(nugget=10.0)
    gp_test.fit(X,y)

    testData = data[data['rng']>=0.9]
    testX = testData[params].values
    y_pred = gp_test.predict(testX)

    resids[i] = y_pred - np.array(testData['Exposure 1'])
    worstPic = np.argmax(np.abs(resids[i]))
    print
    print
    print "Residual: ", resids[i][worstPic]
    print testData.irow(worstPic)



gpResids = np.concatenate([resids[i] for i in range(10)])

linearRegressor = pickle.load(open('mean_Luminance_no_contrast.p'))

print len(gpResids)
print len(linearRegressor.resid)

plt.hist(linearRegressor.resid, color='blue', alpha=0.5, label="Linear", normed=True, bins=np.arange(min(linearRegressor.resid), max(linearRegressor.resid)+50,50))
plt.hist(gpResids, color='green', alpha=0.5, label="GP", normed=True, bins=np.arange(min(gpResids), max(gpResids)+50,50))
plt.legend(loc='upper right')
plt.show()

pickle.dump(gp, open('gp.p', 'wb'))
