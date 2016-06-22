import numpy as np 
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import confusion_matrix
# from multilayer_perceptron import MLPClassifier
w1 = np.loadtxt('iris2.data')


def sigmoidFunc(x, deriv=False):
	if(deriv==True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

def doMLP(X, step = 0.1, momentum=0.05):
	y = w1[:,-1]
	num = y.size
	print "numOfData: ", num 

	feature = w1[0].size -1
	print "numOfFeature: ",feature 

	x = w1[:, 0:feature]
	y=np.reshape(y,(num,1))
	print x.shape, y.shape

	wPathNumber = 10;

	theta0 = 2*np.random.random((wPathNumber, feature,num))-1.0; #w
	theta1 = 2*np.random.random((num,1))-1.0; #v

	print theta0[0].shape
	for i in range(0,1):
		for j in range(0,wPathNumber):
			l0 = x;
			l1 = sigmoidFunc(np.dot(l0, theta0[j] ))#z
			# print "l1.shape: ",l1.shape
			l2 = sigmoidFunc(np.dot(l1, theta1)) #y

			l2_error= (y-l2); 
			# print l2_error.shape
			# print l2
			if(i%100)==0:
				print "Error: "+str(np.mean(np.abs(l2_error)))

			# l2_delta = np.dot(l1,l2_error); #delta v
			# print "l2_delta.shape",l2_delta.shape, np.sum(l2_delta)
			# l1_error = np.dot(l2_delta.T,theta1);
			# print l1_error.shape
			# l1_delta = (l1_error * sigmoidFunc(l1,deriv=True)); #delta w
			l2_delta = l2_error*sigmoidFunc(l2,deriv=True)
			l1_error = l2_delta.dot(theta1.T)
			l1_delta = l1_error * sigmoidFunc(l1,deriv=True)

			theta1 += l1.T.dot(l2_delta)*step + momentum
			theta0[j] += l0.T.dot(l1_delta)*step + momentum
			# if(i%100)==0:
				# print "theta1: ",theta1, "; theta0[j]: ", theta0[j];
	# print y.shape
	getPerformance(l2, y, num)
def SKLearnMLP(w1):
	y = w1[:,-1]
	num = y.size
	print "numOfData: ", num 

	feature = w1[0].size -1
	print "numOfFeature: ",feature 

	x = w1[:, 0:feature]
	print x.shape, y.shape

	a, b = x.shape
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
	# y_predict,y_predictNumber = train_sample(x,y,start,num, (num-start)/10)
	print "y_predict: ",y_predict

	# getPerformance(y_predict, y_cross[start-train_samples:end], y_predictNumber)
	getPerformance(y_predict, y[start-((num-start)/10):num], y_predictNumber)

def train_sample(x, y, start, end, trainSampleNumber):
	print trainSampleNumber
	x_train=x[start:end-trainSampleNumber]
	x_test =x[start-trainSampleNumber:end]
	y_train=y[start:end-trainSampleNumber]
	y_test =y[start-trainSampleNumber:end]
	print "y_test: ", y_test
	num_trainData = x_train.shape[0]
	numfeature = x_train.shape[1]

	# print x_train.shape, x_test.shape, y_train.shape, y_test.shape
	clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(x_train,y_train)
	y_predict = clf.predict(x_test)

	# model = BernoulliRBM(n_components=2)
	# y_predict = model.fit(x_test);
	return y_predict, y_test.size

def crossValidMatrix(x,y,numOfFold):
	numData, numFeature = x.shape;
	c = 1;
	x_new = np.zeros(shape=(numData/numOfFold,numFeature));
	y_new = np.zeros(shape=(numData/numOfFold,1));
	for i in range (0, numData/numOfFold):
		x_new[i] = x[numOfFold*i+c];
		y_new[i] = y[numOfFold*i+c];
	return x_new, y_new


def getPerformance(y_predict, y, dataNum):
	TP1=0; TP2=0;TP3=0;
	FN1=0; FN2=0;FN3=0
	FP1=0; FP2=0; FP3=0;
	TN1=0; TN2=0; TN3=0;
	# print y;
	for i in range(0, dataNum):
		if y_predict[i]==1 and y[i] == 1:
			TP1+=1
		elif y_predict[i] == 2 and y[i] == 2:
			TP2+=1
		elif y_predict[i] == 3 and y[i] == 3:
			TP3+=1

	for i in range(0, dataNum):
		if y_predict[i] != 1 and y[i] != 1:
			TN1+=1
		elif y_predict[i] != 2 and y[i] != 2:
			TN2+=1
		elif y_predict[i] != 3 and y[i] != 3:
			TN3+=1

	for i in range(0, dataNum):
		if y_predict[i] != 1 and y[i] == 1:
			FN1+=1
		elif y_predict[i] != 2 and y[i] == 2:
			FN2+=1
		elif y_predict[i] != 3 and y[i] == 3:
			FN3+=1

	for i in range(0, dataNum):
		if y_predict[i] == 1 and y[i] != 1:
			FP1+=1
		elif y_predict[i] == 2 and y[i] != 2:
			FP2+=1
		elif y_predict[i] == 3 and y[i] != 3:
			FP3+=1

	print "TP: ", TP1, TP2, TP3;
	print "FN: ",FN1, FN2, FN3
	print "FP: ",FP1, FP2, FP3;
	print "TN: ", TN1, TN2, TN3;

	print "----- confusion_matrix: -----"
	print confusion_matrix(y, y_predict);
	print "-----------------------------"
	MeasureFunction("1", TP1, FP1, FN1, TN1, 1);
	MeasureFunction("2", TP2, FP2, FN2, TN2, 1);
	MeasureFunction("3", TP3, FP3, FN3, TN3, 1);

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


doMLP(w1);
# SKLearnMLP(w1)