import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

fig = plt.figure()
df = pd.read_csv(sys.argv[1])
ax = fig.add_subplot(111, projection='3d')
gmin = np.min(df.Gain)
gmax = np.max(df.Gain)
smin = np.min(df.Shutter)
smax = np.max(df.Shutter)

df.Gain = (df.Gain-gmin)/(gmax-gmin)
df.Shutter = (df.Shutter-smin)/(smax-smin)

ax.scatter(df.Gain, df.Shutter, df['ORB Features'])
ax.set_xlabel('Gain (normalized)')
ax.set_ylabel('Shutter (normalized)')
ax.set_zlabel('Features')
"""

plt.scatter(np.log(df.Gain*df.Shutter), df['ORB Features'])
"""
plt.show()
