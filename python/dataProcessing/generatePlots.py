import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from statsmodels.stats.weightstats import ttost_paired

data = pd.read_csv(open('combined_data.csv'))

for t in data.index:
    if int(data.loc[t, 'Baseline']) == 0:
        data.loc[t, 'STF Baseline'] = data.loc[t, 'Succesfully Tracked Features 0']
        data.loc[t, 'STF Experiment'] = data.loc[t, 'Succesfully Tracked Features 1']
    else:
        data.loc[t, 'STF Baseline'] = data.loc[t, 'Succesfully Tracked Features 1']
        data.loc[t, 'STF Experiment'] = data.loc[t, 'Succesfully Tracked Features 0']

pvalue, stats1, stats2 = ttost_paired(data['STF Experiment'], data['STF Baseline'], 0, 10000)

print pvalue
print stats1
print stats2

plt.scatter(data.index, data['STF Baseline'], label='baseline')
plt.scatter(data.index, data['STF Experiment'], color="green", label='experiment')
plt.legend(loc='upper right')
plt.draw()

dataMax = max(data['STF Baseline'].max(), data['STF Experiment'].max())
bins = np.linspace(0, dataMax)

plt.figure()
plt.hist(data['STF Baseline'], alpha = 0.5, bins=bins, label="baseline")
plt.hist(data['STF Experiment'], alpha = 0.5, bins=bins, label="experiment")
plt.legend(loc='upper right')
plt.draw()

plt.figure()
plt.hist(data['STF Experiment'] - data['STF Baseline'], bins=30, color="red")
plt.xlabel('Experiment - Baseline')
plt.show()

