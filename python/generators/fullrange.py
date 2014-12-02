import numpy as np
import pandas as pd

gainRange = np.linspace(16.0, 64.0, 20)
shutterRange = np.linspace(1.0, 531.0, 20)

totalLength = len(gainRange)*len(shutterRange)

gains = pd.Series([0.0]*totalLength)
shutters = pd.Series([0.0]*totalLength)
frames = pd.Series(range(totalLength), name='Frame')

i = 0

for g in gainRange:
    for s in shutterRange:
        gains[i] = g
        shutters[i] = s
        i += 1


df = pd.DataFrame({ 'Gain': gains, 'Shutter': shutters }, index=frames)
df.to_csv('fullrange.csv')
