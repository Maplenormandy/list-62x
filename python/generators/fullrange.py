import numpy as np
import pandas as pd

gainRange = np.linspace(16.0, 64.0, 20)
shutterRange = np.linspace(1.0, 531.0, 20)

totalLength = len(gainRange)*len(shutterRange)

gains = pd.Series([0.0]*totalLength)
shutters = pd.Series([0.0]*totalLength)
frames = pd.Series(range(totalLength), name='Frame')

i = 0
j = 0

for g in gainRange:
    i = 0
    for s in shutterRange:
        # Alternate between increasing and decreasing
        if j % 2 == 0:
            k = i
        else:
            k = len(shutterRange)-1-i
        gains[j*len(gainRange)+k] = g
        shutters[j*len(gainRange)+k] = s
        i += 1
    j += 1


df = pd.DataFrame({ 'Gain': gains, 'Shutter': shutters }, index=frames)
df.to_csv('weavedfullrange.csv')
