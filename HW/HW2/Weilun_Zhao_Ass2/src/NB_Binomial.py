import numpy as np
from sklearn.naive_bayes import MultinomialNB
import math
# w1 = np.loadtxt('spambase2.data')

w1 = np.loadtxt('car2.txt')
numFeature = w1[0].size-1
print "numFeature: ", numFeature

trainData = w1[:,0:numFeature];
# print trainData.shape
y = w1[:,-1];

dataNum = w1[:,0].size
print "data size: ", dataNum

print trainData.shape

featureClass = np.array([4,4,4,3,3,3])
classNum = 4

P_Y = np.zeros(4);


for i in range(0, dataNum):
	if y[i]==1:
		P_Y[0]+=1
	elif y[i]==2:
		P_Y[1]+=1
	elif y[i]==3:
		P_Y[2]+=1
	elif y[i]==4:
		P_Y[3]+=1

print P_Y
P_Y = P_Y/dataNum
print P_Y
maxKind = 4
avgClass = np.zeros([classNum, numFeature,maxKind])
print avgClass[0,0,0]
feature = np.array([[1,2,3,4],[1,2,3,4],[2,3,4,5],[2,4,5],[1,2,3],[1,2,3]])

print feature
def getClassAvg(trainData, y, classNum, p_y,featureClass, numFeature, maxKind,feature):
	avgClass = np.zeros([numFeature, maxKind])
	for i in range(0, numFeature):
		# countFeature = 0.0
		featureKinds = featureClass[i]
		for k in range(0, featureKinds): 
			countFeature = 0
			for j in range(0,dataNum):
				# print "i: ", i, " j: ", " k: ",k," feature[i][k]: ", feature[i][k]," classNum: ",classNum
				if trainData[j][i] == feature[i][k] and y[j] == classNum:
					countFeature += 1;
			avgClass[i, k] = countFeature

	return avgClass


for i in range(0, classNum):
	avgClass[i] = getClassAvg(trainData, y, i+1, P_Y, featureClass, numFeature, maxKind, feature)


# print avgClass
avgProbability=avgClass/dataNum;
# print avgProbability

dictionaryForProb={}
for i in range(0, avgProbability[:,0,0].size):
	for j in range(0,avgProbability[0,:,0].size):
		for k in range(0, avgProbability[0,0,:].size):
			dictionaryForProb[str(i)+str(j)+str(k)] = avgProbability[i][j][k]


def ReturnIndex(feature, theNumFeatureTh, input):
	n = np.array(feature[theNumFeatureTh]);
	length = n.size;
	for i in range(0, length):
		if input == n[i]:
			# return i;
			return str(theNumFeatureTh)+str(i)

ax = ReturnIndex(feature, 0, 2)
# print ax

def getProb(dictionaryForProb, classNumth, feature, theNumFeatureTh, input):
	index = ReturnIndex(feature, theNumFeatureTh, input);
	key = str(classNumth)+index;
	# print key
	return dictionaryForProb[key]

p1_GDA=np.zeros(dataNum);
p2_GDA=np.zeros(dataNum);
p3_GDA=np.zeros(dataNum);
p4_GDA=np.zeros(dataNum);

def getCoef(feature, theNumFeatureTh, inputNum):
	f = np.array(feature[theNumFeatureTh]);
	length = f.size;
	count  =0;
	for i in range(0, length):
		if f[i] == inputNum:
			count =i;
	prob = 0.0 
	prob = math.factorial(length)/math.factorial(count)

	return prob

# print getCoef(feature, 0, 4);

y_GDA = np.zeros(dataNum);

clf = MultinomialNB();
clf.fit(trainData, y);
y_GDA = clf.predict(trainData)


# for i in range(0, dataNum):
# # for i in range(0, 1):
# 	p1_GDA[i] = .0
# 	p2_GDA[i] = .0
# 	p3_GDA[i] = .0
# 	p4_GDA[i] = .0
# 	for j in range(0, numFeature):
# 		prob1 = getProb(dictionaryForProb, 0,feature, j, w1[i][j])
# 		if prob1 != 0:
# 			# p1_GDA[i] += (np.log(P_Y[0])*prob1+ np.log(1.0- P_Y[0])*(1.0-prob1))
# 			# p1_GDA[i] += np.log((P_Y[0])*prob1*(1.0- P_Y[0])*(1.0-prob1))
# 			p1_GDA[i] += np.log(getCoef(feature,j,w1[i][j]))+np.log((P_Y[0])**prob1)+np.log((1.0- P_Y[0])**(1.0-prob1))

# 		prob2 = getProb(dictionaryForProb, 1,feature, j, w1[i][j])
# 		if prob2 != 0:
# 			# p2_GDA[i] += np.log((P_Y[1])*prob2*(1.0- P_Y[1])*(1.0-prob2))
# 			p2_GDA[i] += np.log(getCoef(feature,j,w1[i][j]))+np.log((P_Y[1])**prob2)+np.log((1.0- P_Y[1])**(1.0-prob2))
# 		prob3 = getProb(dictionaryForProb, 2,feature, j, w1[i][j])
# 		if prob3 != 0:
# 			# p3_GDA[i] += np.log((P_Y[2])*prob3*(1.0- P_Y[2])*(1.0-prob3))
# 			p3_GDA[i] += np.log(getCoef(feature,j,w1[i][j]))+np.log((P_Y[2])**prob3)+np.log((1.0- P_Y[2])**(1.0-prob3))
# 		prob4 = getProb(dictionaryForProb, 3,feature, j, w1[i][j])
# 		if prob4 != 0:
# 			# p4_GDA[i] += np.log((P_Y[3])*prob4*(1.0- P_Y[3])*(1.0-prob4))
# 			p4_GDA[i] += np.log(getCoef(feature,j,w1[i][j]))+np.log((P_Y[3])**prob4)+np.log((1.0- P_Y[3])**(1.0-prob4))
# 		# print "i: ", i , " j: " , j, w1[i][j] 

# # # 	print "i",i
# # # 	print p1_GDA[i]
# # # 	print p2_GDA[i]
# 	p1_GDA[i] += np.log(P_Y[0])
# 	p2_GDA[i] += np.log(P_Y[1])
# 	p3_GDA[i] += np.log(P_Y[2])
# 	p4_GDA[i] += np.log(P_Y[3])
# 	if p1_GDA[i] == max(p1_GDA[i],p2_GDA[i],p3_GDA[i],p4_GDA[i] ):
# 		y_GDA[i] = 1
# 	elif p2_GDA[i] == max(p1_GDA[i],p2_GDA[i],p3_GDA[i],p4_GDA[i] ):
# 		y_GDA[i] = 2
# 	elif p3_GDA[i] == max(p1_GDA[i],p2_GDA[i],p3_GDA[i],p4_GDA[i] ):
# 		y_GDA[i] = 3
# 	elif p4_GDA[i] == max(p1_GDA[i],p2_GDA[i],p3_GDA[i],p4_GDA[i] ):
# 		y_GDA[i] = 4

print y_GDA
count = np.zeros(4);
for i in range(0, dataNum):
	if y_GDA[i] == 1:
		count[0]+=1
	elif y_GDA[i] == 2:
		count[1]+=1
	elif y_GDA[i] == 3:
		count[2]+=1
	elif y_GDA[i] == 4:
		count[3]+=1

print count 

# print dictionaryForProb['003']


TP1=0; TP2=0;TP3=0; TP4=0;
FN1=0; FN2=0;FN3=0; FN4=0;
FP1=0; FP2=0; FP3=0; FP4=0; 
TN1=0; TN2=0; TN3=0; TN4=0; 

for i in range(0, dataNum):
	# print "y_GDA: ",i, y_GDA[i]
	# print "y_ri: ",i, y_ri[i]

	if y_GDA[i]==1 and y[i] == 1:
		TP1+=1
	elif y_GDA[i] == 2 and y[i] == 2:
		TP2+=1
	elif y_GDA[i] == 3 and y[i] == 4:
		TP3+=1
	elif y_GDA[i] == 3 and y[i] == 4:
		TP4+=1

for i in range(0, dataNum):
	if y_GDA[i] != 1 and y[i] != 1:
		TN1+=1
	elif y_GDA[i] != 2 and y[i] != 2:
		TN2+=1
	elif y_GDA[i] != 3 and y[i] != 3:
		TN3+=1
	elif y_GDA[i] != 4 and y[i] != 4:
		TN4+=1

for i in range(0, dataNum):
	if y_GDA[i] != 1 and y[i] == 1:
		FN1+=1
	elif y_GDA[i] != 2 and y[i] == 2:
		FN2+=1
	elif y_GDA[i] != 3 and y[i] == 3:
		FN3+=1
	elif y_GDA[i] != 4 and y[i] == 4:
		FN4+=1

for i in range(0, dataNum):
	if y_GDA[i] == 1 and y[i] != 1:
		FP1+=1
	elif y_GDA[i] == 2 and y[i] != 2:
		FP2+=1
	elif y_GDA[i] == 3 and y[i] != 3:
		FP3+=1
	elif y_GDA[i] == 4 and y[i] != 4:
		FP4+=1


print "TP: ", TP1, TP2,TP3, TP4;

print "FN: ",FN1, FN2,FN3, FN4
print "FP: ",FP1, FP2,FP3, FP4
print "TN: ", TN1, TN2, TN3, TN4;
# # # TotTP = TP1+TP2+
# # # TotFN = FN1+FN2+FN3
# # # TotFP = FP1+FP2+FP3
# # # TotTN = TN1+TN2+TN3
# # # print "TotTp: ", TotTP
# # # print "TotFN: ",TotFN 
# # # print "TotFP: ",TotFP ;
# # # print "TotTN: ", TotTN;


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
	precision =0;
	recall = 0
	if tp==0 and fp == 0:
		print  "Precision ", string, 0
	else:
		precision = getPrecision(tp,fp,fn,tn)
		print "Precision ", string, precision;
	if tp==0 and fn ==0:
		print "Recall ",string, 0;
	else:
		recall = getRecall(tp, fp,fn,tn);
		print "Recall ",string, recall;
	if  precision==0 and recall==0 :
		print "Fmeasure ",string, 0
	else:
		print "Fmeasure ",string, getFmeasure(precision,recall,b);

MeasureFunction("1", TP1, FP1, FN1, TN1, 1);
MeasureFunction("2", TP2, FP2, FN2, TN2, 1);
MeasureFunction("3", TP3, FP3, FN3, TN3, 1);
MeasureFunction("4", TP4, FP4, FN4, TN4, 1);


