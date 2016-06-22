import numpy as np
import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
w1 = np.loadtxt('iris2.data')

# print w1.shape

ri = w1[:,0]
print ri.shape

num = ri.size

num1 = 0;
num2 = 0;
num3 = 0;

sum1 = 0;
sum2 = 0;
sum3 = 0;

y_ri = w1[:,-1]

for i in range(0, num):
	if y_ri[i] == 1:
		num1+=1;
		sum1+= ri[i]
	elif y_ri[i] == 2:
		num2+=1;
		sum2+= ri[i]
	elif y_ri[i] == 3:
		num3+=1;
		sum3+= ri[i]

avg1 = sum1/num1
avg2 = sum2/num2
avg3 = sum3/num3

ri_cov = np.cov(ri)
pi = math.pi
p1_GDA=np.zeros(num);
p2_GDA=np.zeros(num);
p3_GDA=np.zeros(num);
print "avg1: ", avg1
print "avg2: ", avg2
print "avg3: ", avg3

print "ri_cov: ", ri_cov
y_GDA = np.zeros(num);
# print "num",num
for i in range(0, num):

	p1_GDA[i] = (1/(np.sqrt(2*pi)*ri_cov))*np.exp(-(ri[i]-avg1)**2/(2*ri_cov**2))

	p2_GDA[i] = (1/(np.sqrt(2*pi)*ri_cov))*np.exp(-(ri[i]-avg2)**2/(2*ri_cov**2))
	p3_GDA[i] = (1/(np.sqrt(2*pi)*ri_cov))*np.exp(-(ri[i]-avg3)**2/(2*ri_cov**2))
	# 

	if p1_GDA[i] == max(p1_GDA[i],p2_GDA[i],p3_GDA[i]):
		y_GDA[i] = 1
	elif p2_GDA[i] == max(p1_GDA[i],p2_GDA[i],p3_GDA[i]):
		y_GDA[i] = 2
	elif p3_GDA[i] == max(p1_GDA[i],p2_GDA[i],p3_GDA[i]):
		y_GDA[i] = 3


# print p1_GDA
# print p2_GDA
# print p3_GDA
# print y_GDA 
# print y_ri

TP1=0; TP2=0;TP3=0;
FN1=0; FN2=0;FN3=0
FP1=0; FP2=0; FP3=0;
TN1=0; TN2=0; TN3=0;

for i in range(0, num):
	# print "y_GDA: ",i, y_GDA[i]
	# print "y_ri: ",i, y_ri[i]

	if y_GDA[i]==1 and y_ri[i] == 1:
		TP1+=1
	elif y_GDA[i] == 2 and y_ri[i] == 2:
		TP2+=1
	elif y_GDA[i] == 3 and y_ri[i] == 3:
		TP3+=1

for i in range(0, num):
	if y_GDA[i] != 1 and y_ri[i] != 1:
		TN1+=1
	elif y_GDA[i] != 2 and y_ri[i] != 2:
		TN2+=1
	elif y_GDA[i] != 3 and y_ri[i] != 3:
		TN3+=1

for i in range(0, num):
	if y_GDA[i] != 1 and y_ri[i] == 1:
		FN1+=1
	elif y_GDA[i] != 2 and y_ri[i] == 2:
		FN2+=1
	elif y_GDA[i] != 3 and y_ri[i] == 3:
		FN3+=1

for i in range(0, num):
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



