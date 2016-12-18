import librosa
from scipy.io import savemat
import os
m = dict()
parent_dir = "/home/pratheeksha/fml/data/IRMAS-TrainingData/" + l

for l in labels:
    m['label'] = l
    for file in os.listdir(parent_dir):
        y, sr = librosa.load(parent_dir +"/"+ file)
        m[file] = librosa.feature.mfcc(y=y, sr=sr)
    savemat("/home/pratheeksha/fml/data/matfiles/"+l+".mat", m)
    print "Processed label" + l


