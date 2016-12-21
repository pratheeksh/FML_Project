#import librosa
#from scipy.io import savemat
import os
parent_dir = "/home/pratheeksha/fml/data/IRMAS-TestingData-Part1/Part1" 
import os
res = dict()
for file in os.listdir(parent_dir):
	if "txt" in file:
		openfile = open(parent_dir +'/'+file, 'r').readlines()
		if len(openfile) > 1:
			continue
		filename = file.split('.')
		filename = '.'.join(filename[:-1])
		for l in openfile:
			l = l.strip()
			res[filename + ".wav"] = l
m = dict()
csvfile = open("testcsv.csv", "w")
for file in res:
	print file, res[file]
#	y, sr = librosa.load(parent_dir +"/"+ file)
 #       m[file] = librosa.feature.mfcc(y=y, sr=sr)
	csvfile.write(file + ',' + res[file] + '\n')
#savemat("/home/pratheeksha/fml/data/matfiles/test.mat", m)


