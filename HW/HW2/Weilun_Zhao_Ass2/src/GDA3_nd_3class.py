import numpy as np

import math
# w1 = np.loadtxt('spambase2.data')

w1 = np.loadtxt('iris2.data')
numFeature = w1[0,:].size-1
print "numFeature: ", numFeature

trainData = w1[:,0:numFeature];
# print trainData.shape
dataNum = w1[:,0].size
print "data size: ", dataNum

diffTypeNum = 3;
y_ri = w1[:,-1]

sumClass = np.zeros((diffTypeNum, numFeature),dtype=np.float);
avgClass = np.zeros((diffTypeNum, numFeature),dtype=np.float);

sumClassCount = np.zeros(diffTypeNum,dtype=np.float);

for i in range(0, dataNum):
	if y_ri[i] == 1:
		sumClass[0]+= trainData[i];
		sumClassCount[0] += 1;
	elif y_ri[i] == 2:
		sumClass[1]+= trainData[i];
		sumClassCount[1] += 1;
	elif y_ri[i] == 3:
		sumClass[2]+= trainData[i];
		sumClassCount[2] += 1;

for i in range(0,diffTypeNum):
	avgClass[i] = sumClass[i]/sumClassCount[i]

print avgClass

pi = math.pi

ri_cov = np.matrix(np.cov(np.matrix(trainData).T))
ri_cov_I = ri_cov.I
valueOfRi_cov = np.linalg.det(ri_cov);

p1_GDA=np.zeros(dataNum);
p2_GDA=np.zeros(dataNum);
p3_GDA=np.zeros(dataNum);

y_GDA = np.zeros(dataNum);

for i in range(0, dataNum):
	p1_GDA[i] = (1/(np.sqrt((2*pi)**dataNum*valueOfRi_cov)))*np.exp((-1/2)*((np.matrix(trainData[i]-avgClass[0]))*ri_cov_I*np.matrix(trainData[i]-avgClass[0]).T))
	p2_GDA[i] = (1/(np.sqrt((2*pi)**dataNum*valueOfRi_cov)))*np.exp((-1/2)*((np.matrix(trainData[i]-avgClass[1]))*ri_cov_I*np.matrix(trainData[i]-avgClass[1]).T))
	p3_GDA[i] = (1/(np.sqrt((2*pi)**dataNum*valueOfRi_cov)))*np.exp((-1/2)*((np.matrix(trainData[i]-avgClass[2]))*ri_cov_I*np.matrix(trainData[i]-avgClass[2]).T))
	
	if p1_GDA[i] == max(p1_GDA[i],p2_GDA[i],p3_GDA[i]):
		y_GDA[i] = 1
	elif p2_GDA[i] == max(p1_GDA[i],p2_GDA[i],p3_GDA[i]):
		y_GDA[i] = 2
	elif p3_GDA[i] == max(p1_GDA[i],p2_GDA[i],p3_GDA[i]):
		y_GDA[i] = 3

# print y_GDA

TP1=0; TP2=0;TP3=0;
FN1=0; FN2=0;FN3=0
FP1=0; FP2=0; FP3=0;
TN1=0; TN2=0; TN3=0;

for i in range(0, dataNum):
	# print "y_GDA: ",i, y_GDA[i]
	# print "y_ri: ",i, y_ri[i]

	if y_GDA[i]==1 and y_ri[i] == 1:
		TP1+=1
	elif y_GDA[i] == 2 and y_ri[i] == 2:
		TP2+=1
	elif y_GDA[i] == 3 and y_ri[i] == 3:
		TP3+=1

for i in range(0, dataNum):
	if y_GDA[i] != 1 and y_ri[i] != 1:
		TN1+=1
	elif y_GDA[i] != 2 and y_ri[i] != 2:
		TN2+=1
	elif y_GDA[i] != 3 and y_ri[i] != 3:
		TN3+=1

for i in range(0, dataNum):
	if y_GDA[i] != 1 and y_ri[i] == 1:
		FN1+=1
	elif y_GDA[i] != 2 and y_ri[i] == 2:
		FN2+=1
	elif y_GDA[i] != 3 and y_ri[i] == 3:
		FN3+=1

for i in range(0, dataNum):
	if y_GDA[i] == 1 and y_ri[i] != 1:
		FP1+=1
	elif y_GDA[i] == 2 and y_ri[i] != 2:
		FP2+=1
	elif y_GDA[i] == 3 and y_ri[i] != 3:
		FP3+=1


print "TP: ", TP1, TP2, TP3;
print "FN: ",FN1, FN2, FN3
print "FP: ",FP1, FP2, FP3;
print "TN: ", TN1, TN2, TN3;
TotTP = TP1+TP2+TP3
TotFN = FN1+FN2+FN3
TotFP = FP1+FP2+FP3
TotTN = TN1+TN2+TN3
print "TotTp: ", TotTP
print "TotFN: ",TotFN 
print "TotFP: ",TotFP ;
print "TotTN: ", TotTN;


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
MeasureFunction("3", TP3, FP3, FN3, TN3, 1);
MeasureFunction("tot", TotTP, TotFP, TotFN, TotTN, 1);

