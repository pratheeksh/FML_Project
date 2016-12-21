testdata = open("testcsv.csv").readlines()
submission = open("logs/submission.csv").readlines()
classes = ['cla', 'gac', 'org', 'sax', 'vio', 'cel', 'flu', 'gel', 'pia', 'tru', 'voi']
count = 0
labeldict = dict()

for l in testdata:
	label = l.split(',')[-1]
	labeldict[','.join(l.split(',')[:-1])]= label
for l in submission:
	label = l.split(',')[-1]
	label = int(label) - 1
	key = ','.join(l.split(',')[:-1])
	if key in labeldict:
		actual = labeldict[key]
		if actual.strip() == classes[label]:
			count += 1

print count, float(count)/len(submission)
