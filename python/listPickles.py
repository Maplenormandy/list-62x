import os
import pickle

for f in os.listdir('.'):
    if f[-2:] == '.p':
        print f
        print pickle.load(open(f)).summary()
        print
        print '------------------------------'
        print
