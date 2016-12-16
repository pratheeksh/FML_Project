import os
labels = ['cla', 'gac', 'org', 'sax', 'vio', 'cel', 'flu', 'gel', 'pia', 'tru', 'voi']
filename = open("labels.csv", "w")
for l in labels:
    parent_dir = "/home/pratheeksha/fml/data/IRMAS-TrainingData/" + l
    m = dict()
    m['label'] = l
    for file in os.listdir(parent_dir):
    	filename.write(file +',' + l + '\n')	
    print "Processed label" + l


