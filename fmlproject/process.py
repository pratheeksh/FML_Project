import librosa
from scipy.io import savemat
import os
labels = ['cla', 'gac', 'org', 'sax', 'vio', 'cel', 'flu', 'gel', 'pia', 'tru', 'voi']

for l in labels:
    parent_dir = "/Users/pratheeksha/School/fml/data/IRMAS-TrainingData/" + l
    m = dict()
    m['label'] = l
    for file in os.listdir(parent_dir):
        y, sr = librosa.load(parent_dir +"/"+ file)
        m[file] = librosa.feature.mfcc(y=y, sr=sr)
    savemat("/Users/pratheeksha/School/FML/data/matfiles"+file+".mat", m)
    print "Processed label" + l


