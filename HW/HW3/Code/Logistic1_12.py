import numpy as np
import math
from scipy.misc import imread
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
# w1 = imread("testImage.jpg")
w1 = np.loadtxt('spambase2.txt')
	
def doLogist(w1):
	y = w1[:,-1]
	num = y.size
	print "numOfData: ", num 

	feature = w1[0].size -1
	print "numOfFeature: ",feature 

	x = w1[:, 0:feature]
	print x.shape, y.shape

	a, b = x.shape
	# a, b, c = x.shape
	# print a, b
	x_cross, y_cross = crossValidMatrix(x,y,2);
	# print y_cross

	start = 0;
	end = y_cross.size;
	# start = 0;
	# end = y.size;
	# for i in range(0,10):

	train_samples = (end-start)/10

	# print "s,e,t: ", start, end, train_samples
	# x_cross = NonLinearCombations(x_cross)
	# x = NonLinearCombations(x);

	y_predict,y_predictNumber = train_sample(x_cross,y_cross,start,end, train_samples)
	# print y_predict
	# y_predict,y_predictNumber = train_sample(x,y,start,num, (start-num)/10)

	# getPerformance(y_predict, y_cross[start-train_samples:end], y_predictNumber)
	getPerformance(y_predict, y[start-((start-num)/10):num], y_predictNumber)

def crossValidMatrix(x,y,numOfFold):
	numData, numFeature = x.shape;
	c = 0;
	x_new = np.zeros(shape=(numData/numOfFold,numFeature));
	y_new = np.zeros(shape=(numData/numOfFold,1));
	for i in range (0, numData/numOfFold):
		x_new[i] = x[numOfFold*i+c];
		y_new[i] = y[numOfFold*i+c];
	return x_new, y_new

def NonLinearCombations(x):
	numData, numFeature = x.shape;
	# print x.shape
	x_new = np.zeros(shape=(numData, 2*numFeature-1))
	for i in range(0, numData):
		for j in range(1, numFeature):
			x_new[i][j] = x[i][j];
		for k in range(1, numFeature-1):
			x_new[i][numFeature + k] = x[i][k]*x[i][k-1];

	# print x_new[0]
	return x_new
def train_sample(x, y, start, end, trainSampleNumber):
	x_train=x[start:end-trainSampleNumber]
	x_test =x[start-trainSampleNumber:end]
	y_train=y[start:end-trainSampleNumber]
	y_test =y[start-trainSampleNumber:end]

	num_trainData = x_train.shape[0]
	numfeature = x_train.shape[1]
	# print x_train.shape, x_test.shape, y_train.shape, y_test.shape


	lr = LogisticRegression()
	lr.fit(x_train,y_train)
	y_predict = lr.predict(x_test)
	# print y_predict

	theta = np.zeros(shape=(numfeature,1))
	alpha = 1

	# num_iteration = 500;
	# for i in range(0, num_iteration):
	# 	# print "theta[0]: ",theta[0]
	# 	theta = getNewTheta(theta, alpha, x_train, y_train, num_trainData)
	# print theta.shape
	# print "getCostFunction:", getCostFunction(theta, x_test, y_test);
	# print getPrediction(theta,x_test, y_test);
	# y_predict = getPrediction(theta,x_train, y_train)
	# print y_predict
	return y_predict, y_test.size
def sigmoid(X):
	# print X
	return 1.0/(1.0+np.exp(-X))

def getHyphosis(X):
	if X > 0.5:
		return 1;
	else:
		return 0;

def getNewTheta(theta, alpha, X, Y, numData):
	numFeature = theta.size;
	# print "numfeature:",numFeature
	newTheta = np.zeros(shape=(numFeature,1));
	newHandle = np.zeros(shape=(numFeature,1));
	# newHandle +=((sigmoid((theta.T).dot(X[0]))-Y[0])*X[0]).reshape(numFeature,1)
	# print newHandle.shape


	for i in range(0, numData):
		newHandle+=((( getHyphosis(sigmoid((theta.T).dot(X[i])))-Y[i])*X[i])).reshape(numFeature,1)
	newTheta = theta - (alpha)*(1.0/numData)*newHandle;
	# print newTheta
	return newTheta;

def getCostFunction(theta, x_test, y_test):
	numData = x_test.shape[0]
	numFeature = x_test.shape[1]
	J = 0.0;
	# print "theta*dot: ",(1.0/numData)*((-y_test[0]) * np.log(sigmoid((theta.T).dot(x_test[0]))) - (1-y_test[0]) * np.log(1 - sigmoid((theta.T).dot(x_test[0])) ) )
	# print "cost function:", (-y_test[0])*math.log(sigmoid( (theta.T).dot(x_test[0]) )) - (1-y_test[0])*math.log(1-sigmoid( (theta.T).dot(x_test[0]) ))
	# print  "log:", math.log(sigmoid((theta.T).dot(x_test[0])))
	J += (1.0/numData)*( (y_test[0]) * np.log( sigmoid((theta.T).dot(x_test[0])) ) + (1.0-y_test[0]) * np.log(1 - sigmoid((theta.T).dot(x_test[0])) ) )

	return J;

def getPrediction(theta, x_test, y_test):
	numData = x_test.shape[0]
	numFeature = x_test.shape[1]
	y_predict = np.zeros(shape=(numData,1))
	for i in range(0, numData):
		print sigmoid((theta.T).dot(x_test[i]))
	# 	y_predict[i] = getHyphosis( sigmoid((theta.T).dot(x_test[i])) )
	return y_predict

def getPerformance(y_predict, y, dataNum):
	TP1=0; TP2=0;
	FN1=0; FN2=0;
	FP1=0; FP2=0; 
	TN1=0; TN2=0; 
	# print y;
	for i in range(0, dataNum):
		if y_predict[i]==1 and y[i] == 1:
			TP1+=1
		elif y_predict[i] == 0 and y[i] == 0:
			TP2+=1

	for i in range(0, dataNum):
		if y_predict[i] != 1 and y[i] != 1:
			TN1+=1
		elif y_predict[i] != 0 and y[i] != 0:
			TN2+=1


	for i in range(0, dataNum):
		if y_predict[i] != 1 and y[i] == 1:
			FN1+=1
		elif y_predict[i] != 0 and y[i] == 0:
			FN2+=1

	for i in range(0, dataNum):
		if y_predict[i] == 1 and y[i] != 1:
			FP1+=1
		elif y_predict[i] == 0 and y[i] != 0:
			FP2+=1


	print "TP: ", TP1, TP2;
	print "FN: ",FN1, FN2
	print "FP: ",FP1, FP2;
	print "TN: ", TN1, TN2;
	MeasureFunction("1", TP1, FP1, FN1, TN1, 1);
	MeasureFunction("2", TP2, FP2, FN2, TN2, 1);
	# print confusion_matrix(y, y_predict);
	print " ******************************* "

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

	

doLogist(w1)

# print w1[0].reshape(4,1)