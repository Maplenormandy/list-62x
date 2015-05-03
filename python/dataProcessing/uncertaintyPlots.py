import pandas as pd
import matplotlib.pyplot as plt
import imageFeatures as imf
import cv2
import pickle
import numpy as np
import os.path

from statsmodels.stats.weightstats import ttost_paired

data = pd.read_csv(open('bldg33_2015-04-18_rawdata.csv'), index_col='Frame')
gp = pickle.load(open('gp.p','r'))

if os.path.isfile('bldg33_2015-04-18_data.csv'):
    data = pd.read_csv(open('bldg33_2015-04-18_data.csv'), index_col='Frame')
else:

    surf = cv2.SURF()

    currentRun = -1

    for t in data.index:
        print t

        if int(data.loc[t, 'Baseline']) == 0:
            data.loc[t, 'STF Baseline'] = data.loc[t, 'Succesfully Tracked Features 0']
            data.loc[t, 'STF Experiment'] = data.loc[t, 'Succesfully Tracked Features 1']
            data.loc[t, 'Exposure Baseline'] = imf.settingsToExposure(data.loc[t, 'Shutter 0'], data.loc[t, 'Gain 0'])
            data.loc[t, 'Exposure Experiment'] = imf.settingsToExposure(data.loc[t, 'Shutter 1'], data.loc[t, 'Gain 1'])
            filename = '../data/bldg33_2015-04-18_0/' + data.loc[t, 'Image File 1']
        else:
            data.loc[t, 'STF Baseline'] = data.loc[t, 'Succesfully Tracked Features 1']
            data.loc[t, 'STF Experiment'] = data.loc[t, 'Succesfully Tracked Features 0']
            data.loc[t, 'Exposure Baseline'] = imf.settingsToExposure(data.loc[t, 'Shutter 1'], data.loc[t, 'Gain 1'])
            data.loc[t, 'Exposure Experiment'] = imf.settingsToExposure(data.loc[t, 'Shutter 0'], data.loc[t, 'Gain 0'])
            filename = '../data/bldg33_2015-04-18_0/' + data.loc[t, 'Image File 0']

        frame = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
        kp, desc = surf.detectAndCompute(frame, None)

        data.loc[t, 'Contrast'] = imf.contrast(frame)
        data.loc[t, 'Mean Lum'] = imf.meanLuminance(frame)
        data.loc[t, 'FG Illum'], data.loc[t, 'BG Illum'] = imf.weightedLuminance(frame)
        data.loc[t, 'Blur Lum'] = imf.gaussianBlurfeatureLuminance(frame, kp)

        if data.loc[t, 'Run Number'] > currentRun:
            currentRun = data.loc[t, 'Run Number']
        else:
            testX = np.array([oldExposure,
                oldContrast, data.loc[t, 'Contrast'],
                oldBlurLum, data.loc[t, 'Blur Lum'],
                oldFg, oldBg,
                data.loc[t, 'FG Illum'], data.loc[t, 'BG Illum']])
            y_pred, sigma2_pred = gp.predict(testX, eval_MSE=True)
            data.loc[t, 'Exposure Sigma'] = np.sqrt(sigma2_pred)
            data.loc[t, 'Exposure Pred'] = y_pred

        oldExposure = data.loc[t, 'Exposure Experiment']
        oldContrast = data.loc[t, 'Contrast']
        oldMeanLum = data.loc[t, 'Mean Lum']
        oldFg = data.loc[t, 'FG Illum']
        oldBg = data.loc[t, 'BG Illum']
        oldBlurLum = data.loc[t, 'Blur Lum']

    data.to_csv('bldg33_2015-04-18_data.csv')

data['Timestamp'] = pd.to_datetime(data['Timestamp'])

run0 = data[data['Run Number'] == 0]
data.drop(run0.index[:1])
run0 = run0[1:]
run1 = data[data['Run Number'] == 1]
data.drop(run1.index[:1])
run1 = run1[1:]


pvalue, stats1, stats2 = ttost_paired(data['STF Experiment'], data['STF Baseline'], 0, 10000)

print pvalue
print stats1
print stats2

fig, ax1 = plt.subplots()
ax1.plot(run0['Timestamp'], run0['STF Baseline'], '.', color=(0, 1, 0))
ax1.plot(run0['Timestamp'], run0['STF Experiment'], '.', color=(1, 0, 0))
ax1.plot(run1['Timestamp'], run1['STF Baseline'], '.', color=(0, 1, 0))
ax1.plot(run1['Timestamp'], run1['STF Experiment'], '.', color=(1, 0, 0))

ax2 = ax1.twinx()
ax2.plot(run0['Timestamp'], run0['Exposure Baseline'], '-', color=(0, 0.75, 0))
ax2.plot(run0['Timestamp'], run0['Exposure Experiment'], '-', color=(0.75, 0, 0))

expExp = np.array(run0['Exposure Experiment'])
expSig = np.array(run0['Exposure Sigma'])
time = np.array([float(x) for x in np.array(run0['Timestamp'])])

ax2.fill_between(time, expExp - expSig, expExp + expSig, alpha = 0.5)

expExp = np.array(run1['Exposure Experiment'])
expSig = np.array(run1['Exposure Sigma'])
time = np.array([float(x) for x in np.array(run1['Timestamp'])])

ax2.fill_between(time, expExp - expSig, expExp + expSig, alpha = 0.5)



ax2.plot(run1['Timestamp'], run1['Exposure Baseline'], '-', color=(0, 0.75, 0))
ax2.plot(run1['Timestamp'], run1['Exposure Experiment'], '-', color=(0.75, 0, 0))

ax1.set_ylabel('Number of Successfully Tracked Features')
ax2.set_ylabel('Exposure')
ax1.set_xlabel('Timestamp')

plt.show()

