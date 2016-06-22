import numpy as np
from sklearn.naive_bayes import BernoulliNB
import math
# w1 = np.loadtxt('spambase2.data')

w1 = np.loadtxt('spambase2.txt')
numFeature = w1[0].size-1
print "numFeature: ", numFeature

trainData = w1[:,0:numFeature];
# print trainData.shape
y = w1[:,-1];

dataNum = w1[:,0].size
print "data size: ", dataNum

print trainData.shape

for i in range(0, dataNum):
	for j in range(0, numFeature):
		if trainData[i][j]!=0:
			trainData[i][j] = 1

for i in range(0, dataNum):
	for j in range(0, numFeature):
		if w1[i][j]!=0:
			w1[i][j] = 1

print w1.shape
# print trainData
avg1 = np.zeros(numFeature, dtype = float);
avg2 = np.zeros(numFeature, dtype = float);

# print avg1[0]
def getProbabilityEqualForEachFeature(w, x):
	numFeature = w[0].size -1;
	aveOneFeature=np.zeros(numFeature, dtype = float);
	# print aveOneFeature.shape
	y = w[:, -1]
	numDate = y.size
	numFor1 = 0.0;
	for i in range(0, numDate):
		if y[i]==x:
			numFor1+=1;

	# print numFor1;

	for i in range(0, numFeature):
		count1Times = 0.0;
		for j in range(0, numDate):
			# print "w", i,j,w[i][j]
			if y[j] == x and w[j][i] == x:
				count1Times+=1;
		# print i, count1Times
		aveOneFeature[i] = count1Times/numFor1
		if aveOneFeature[i] == 1:
			aveOneFeature[i]-=0.01
		elif aveOneFeature[i] == 0:
			aveOneFeature[i]+=0.01
	return numFor1, aveOneFeature
numOfClass1 = 0;
numOfClass1, avg1 = getProbabilityEqualForEachFeature(w1,1);
numOfClass2, avg2 = getProbabilityEqualForEachFeature(w1,0);

print numOfClass1
print numOfClass2

p1_GDA=np.zeros(dataNum);
p2_GDA=np.zeros(dataNum);
pX1=1.0
pX2=1.0
#get p(x1,...,x57)
for i in range(0,57):
	pX1*= avg1[i];
	pX2*= avg2[i];

pY1 = numOfClass1/dataNum
pY2 = numOfClass2/dataNum

print pY1

y_GDA = np.zeros(dataNum);
# threshold = (0.5)

# for i in range(0, dataNum):
# 	p1_GDA[i] = 1.0
# 	p2_GDA[i] = 1.0
# 	for j in range(0, numFeature):
# 		# p1_GDA[i] *= avg1[j]*w1[i][j]+(1-avg1[j])*(1.0-w1[i][j])
# 		# p2_GDA[i] *= avg2[j]*w1[i][j]+(1-avg2[j])*(1.0-w1[i][j])
# 		p1_GDA[i] += (np.log(pY1)*w1[i][j] + np.log(1.0-pY1)*(1.0-w1[i][j]))
# 		p2_GDA[i] += (np.log(pY2)*w1[i][j] + np.log(1.0-pY2)*(1.0-w1[i][j]))
# 		# p1_GDA[i] += (np.log(pY1)*avg1[j] + np.log(1.0-pY1)*(1.0-avg1[j]))
# 		# p2_GDA[i] += (np.log(pY2)*avg2[j] + np.log(1.0-pY2)*(1.0-avg2[j]))
# 	# p1_GDA[i] = p1_GDA[i]/numFeature
# 	# p2_GDA[i] = 1 - p1_GDA[i]
# 	# p1_GDA[i] *= pY1
# 	# p2_GDA[i] *= pY2
# 	print "i",i
# 	print p1_GDA[i]
# 	print p2_GDA[i]
# 	if p1_GDA[i] == max(p1_GDA[i],p2_GDA[i]):
# 		y_GDA[i] = 1
# 	elif p2_GDA[i] == max(p1_GDA[i],p2_GDA[i]):
# 		y_GDA[i] = 0
clf = BernoulliNB();
clf.fit(trainData, y);
y_GDA = clf.predict(trainData)

print y_GDA

s = y_GDA.size
count =0
for i in range(0, s):
	if y_GDA[i] == 1:
		count+=1;

print count


TP1=0; TP2=0;
FN1=0; FN2=0;
FP1=0; FP2=0; 
TN1=0; TN2=0; 

for i in range(0, dataNum):
	# print "y_GDA: ",i, y_GDA[i]
	# print "y_ri: ",i, y_ri[i]

	if y_GDA[i]==1 and y[i] == 1:
		TP1+=1
	elif y_GDA[i] == 0 and y[i] == 0:
		TP2+=1

for i in range(0, dataNum):
	if y_GDA[i] != 1 and y[i] != 1:
		TN1+=1
	elif y_GDA[i] != 0 and y[i] != 0:
		TN2+=1


for i in range(0, dataNum):
	if y_GDA[i] != 1 and y[i] == 1:
		FN1+=1
	elif y_GDA[i] != 0 and y[i] == 0:
		FN2+=1


for i in range(0, dataNum):
	if y_GDA[i] == 1 and y[i] != 1:
		FP1+=1
	elif y_GDA[i] == 0 and y[i] != 0:
		FP2+=1


print "TP: ", TP1, TP2;
print "FN: ",FN1, FN2
print "FP: ",FP1, FP2;
print "TN: ", TN1, TN2;
# TotTP = TP1+TP2+
# TotFN = FN1+FN2+FN3
# TotFP = FP1+FP2+FP3
# TotTN = TN1+TN2+TN3
# print "TotTp: ", TotTP
# print "TotFN: ",TotFN 
# print "TotFP: ",TotFP ;
# print "TotTN: ", TotTN;


def getAccurcy(tp, fp,fn,tn):
	return (float)(tp+tn)/(tp+fp+fn+tn)

def getPrecision(tp,fp,fn,tn):
	return (float)(tp)/(tp+fp)

def getRecall(tp,fp,fn,tn):
	return (float)(tp)/(tp+fn)

def getFmeasure(precision, recall, b):
	return (float)(1+b**2)*(precision*recall)/(precision+recall)

def MeasureFunction(string,tp,fp,fn,tn,b):
	print "Accurcy ",string, getAccurcy(tp,fp,fn,tn);
	precision = getPrecision(tp,fp,fn,tn)
	print "Precision ", string, precision;
	recall = getRecall(tp, fp,fn,tn);
	print "Recall ",string, recall;
	print "Fmeasure ",string, getFmeasure(precision,recall,b);

MeasureFunction("1", TP1, FP1, FN1, TN1, 1);
MeasureFunction("2", TP2, FP2, FN2, TN2, 1);
