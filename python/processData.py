import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

df = pd.read_csv(sys.argv[1])
ax.scatter(df.Gain, df.Shutter, df['ORB Features'])

plt.show()
