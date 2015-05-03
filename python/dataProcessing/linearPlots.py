import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv(open('optimalPointsWmask_16s.csv'))
data['Exposure 0'] = data['Shutter 0']*data['Gain 0']/16.0
data['Exposure 1'] = data['Shutter 1']*data['Gain 1']/16.0

data['Exposure 1 Pred'] = np.clip(111.2148 + 0.6940*data['Exposure 0'] - 2.7011*data['Mean 0'] + 2.6972*data['Mean 1'], 0.0, 2000.0)

plt.scatter(data['Exposure 1'], data['Exposure 1 Pred'] - data['Exposure 1'])
plt.xlabel('Best Exposure')
plt.ylabel('Residual')

plt.show()
