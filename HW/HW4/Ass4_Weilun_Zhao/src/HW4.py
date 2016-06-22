import numpy as np 
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
from sklearn import svm
from cvxopt import matrix, solvers
from sklearn.metrics import confusion_matrix

def generateLSdData():
	n = 50; #data size (50*2, 2)
	a = np.random.random([2])
	for i in range(1, n):
		a = np.row_stack((a,np.random.random([2])))

	a = np.column_stack((a, np.zeros(n).reshape(n,1)-1)) 

	# print a
	b = -np.random.random([2])
	for i in range(1, n):
		b = np.row_stack((b,-np.random.random([2])))

	b = np.column_stack((b, np.zeros(n).reshape(n,1)+1))

	c = np.row_stack((a,b))
	# print c.shape
	dataFile = open("dataFile.txt", "w+");
	# dataFile = TemporaryFile()
	np.save(dataFile,c);

	dataFile.seek(0)
	print np.load(dataFile)
	print "create data matrix:", c.shape

def generateNonLSdData():
	n = 50; #data size (n*2, 2)
	a = np.random.random([2])-0.5
	for i in range(1, n):
		a = np.row_stack((a,np.random.random([2])-0.5))
	a = np.column_stack((a, np.zeros(n).reshape(n,1)-1)) 

	b = np.random.random([2])-0.5
	for i in range(1, n):
		b = np.row_stack((b,np.random.random([2])-0.5))

	b = np.column_stack((b, np.zeros(n).reshape(n,1)+1))

	c = np.row_stack((a,b))
	dataFile = open("dataFileNonLS.txt", "w+");
	np.save(dataFile,c);

def plotData(gen):
	num, col = gen.shape
	# print col
	oneType = num/2;
	x1 = gen[:oneType, [0]]
	y1 = gen[:oneType, [1]]
	x2 = gen[oneType:, [0]]
	y2 = gen[oneType:, [1]]

	plt.plot(x1, y1, 'ro')
	plt.plot(x2, y2, 'b*')
	# plt.show()

def loadData(filename):
	dataFile = open(filename, "r");

	dataFile.seek(0)
	c = np.load(dataFile)
	print "data Info:", c.shape
	return c

def crossValidMatrix(x,y,numOfFold, c):
	numData, numFeature = x.shape;
	# c = 0;
	x_new = np.zeros(shape=(numData/numOfFold,numFeature));
	y_new = np.zeros(shape=(numData/numOfFold,1));
	for i in range (0, numData/numOfFold):
		x_new[i] = x[numOfFold*i+c];
		y_new[i] = y[numOfFold*i+c];
	return x_new, y_new

def SVM_SK_Linear(gen, testNum):
	num, col = gen.shape

	x = gen[:,:2]
	y = gen[:,-1:]

	# x_train = gen[:num - testNum,:2]
	# y_train = gen[:num - testNum,-1:]

	# x_test = gen[-testNum:, :2]
	# y_test = gen[-testNum:, -1:]
	fold = 2
	x_train, y_train = crossValidMatrix(x, y, fold, 1);
	x_test, y_test = crossValidMatrix(x,y,fold,0);
	# print x_train.shape, y_train.shape, x_test.shape, y_test.shape
	# clf = svm.SVC(kernel='linear',C=1)

	clf = svm.SVC(kernel='linear',C=1)
	clf.fit(x_train, y_train)

	y_predict = clf.predict(x_test)
	print "predict: ", y_predict

	# predict = clf.predict(x_train)
	# print predict
	# print num
	getPerformance(y_predict, y_test, num/fold);

	w = clf.coef_[0]
	# print w[0], w[1]
	a = -w[0] / w[1]

	print "a: ",a
	xx = np.linspace(-2, 2)
	yy = a*xx - (clf.intercept_[0])/w[1]

	# margin = 1/np.sqrt(np.sum(clf.coef_ ** 2))
	# yy_down = yy + a*margin;
	# yy_up = yy - a*margin;

	b=clf.support_vectors_[0]
	yy_down = a*xx+(b[1]-a*b[0])

	b=clf.support_vectors_[-1]
	yy_up = a*xx+(b[1]-a*b[0])

	# print clf.support_vectors_[0]
	# plt.figure(1, figsize =(4,3))
	plt.clf()
	plt.plot(xx, yy, 'k-')
	plt.plot(xx, yy_down, 'k--')
	plt.plot(xx, yy_up, 'k--')
	plt.axis([-2, 2, -2, 2])
	plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1],s=80, facecolors='none',zorder=10)
	
	# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                # facecolors='none', zorder=10)
	# plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, zorder=10, cmap=plt.cm.Paired)
def SVM(gen, testNum, C=1.0):
	num, col = gen.shape
	trainNum = num - testNum;

	x_train = gen[:trainNum ,:2]
	y_train = gen[:trainNum ,-1:]

	x_test = gen[-testNum:, :2]
	y_test = gen[-testNum:, -1:]

	feature = col - 1;
	K = np.zeros((trainNum, trainNum))

	for i in range(trainNum):
		for j in range(trainNum):
			K[i,j] = np.dot(x_train[i], x_train[j])

	P = matrix(np.outer(y_train, y_train)*K)
	# print P
	q = matrix(np.ones(trainNum)*-1)
	A = matrix(y_train, (1, trainNum))
	b = matrix(0.0)

	#  hard margin:
	# G = matrix(np.diag(np.ones(trainNum) * -1))
	# h = matrix(np.zeros(trainNum))

	# soft margin:
	tmp1 = np.diag(np.ones(trainNum)*-1)
	tmp2 = np.identity(trainNum)
	G = matrix(np.vstack((tmp1, tmp2)))
	tmp1 = np.zeros(trainNum)
	tmp2 = np.ones(trainNum)* C 
	h = matrix(np.hstack((tmp1,tmp2)))

	solution = solvers.qp(P,q,G,h,A,b)
	a = np.ravel(solution['x'])
	# print a.shape
	w = np.zeros([2])
	# print w
	for i in range(trainNum):
		w+=a[i]*y_train[i]*x_train[i]

	# print w

	w0 = 0
	for i in range(trainNum):
		w0+= y_train[i]+np.dot(w.T, x_train[i]);

	w0 = w0/trainNum;
	# print w0

	slop = -w[0] / w[1]
	xx = np.linspace(-1.5, 1.5)
	yy = slop*xx - (w0)/w[1]


	plt.plot(xx, yy, 'k-')
	def f(x, w, b, c=0):
		return (-w[0] * x - b + c) / w[1]
	a0 = -1.2; a1 = f(a0, w, w0, 1)
	b0 = 1.2; b1 = f(b0, w, w0, 1)
	plt.plot([a0,b0], [a1,b1], "k--")

	a0 = -1.2; a1 = f(a0, w, w0, -1)
	b0 = 1.2; b1 = f(b0, w, w0, -1)
	plt.plot([a0,b0], [a1,b1], "k--")

def GaussianKernel(x1, x2, sigma=0.1):
	d = Euclidean_distance(x1, x2);
	return np.exp(-(d**2)/(2*sigma**2))

def Euclidean_distance(x1, x2):
	dis = np.dot((x1-x2).T,(x1-x2))
	# print dis
	return dis
def SVM_Kernel(gen, testNum, C=1.0):
	num, col = gen.shape
	trainNum = num - testNum;

	x_train = gen[:trainNum ,:2]
	y_train = gen[:trainNum ,-1:]

	x_test = gen[-testNum:, :2]
	y_test = gen[-testNum:, -1:]

	feature = col - 1;
	K = np.zeros((trainNum, trainNum))

	for i in range(trainNum):
		for j in range(trainNum):
			K[i, j] = GaussianKernel(x_train[i], x_train[j])

	# print "GaussianKernel: ",GaussianKernel(x_train[0],x_train[1])

	P = matrix(np.outer(y_train, y_train)*K)
	# print P
	q = matrix(np.ones(trainNum)*-1)
	A = matrix(y_train, (1, trainNum))
	b = matrix(0.0)

	#  hard margin:
	# G = matrix(np.diag(np.ones(trainNum) * -1))
	# h = matrix(np.zeros(trainNum))

	# soft margin:
	tmp1 = np.diag(np.ones(trainNum)*-1)
	tmp2 = np.identity(trainNum)
	G = matrix(np.vstack((tmp1, tmp2)))
	tmp1 = np.zeros(trainNum)
	tmp2 = np.ones(trainNum)* C 
	h = matrix(np.hstack((tmp1,tmp2)))

	solution = solvers.qp(P,q,G,h,A,b)
	a = np.ravel(solution['x'])
	# print a.shape
	w = np.zeros([2])
	# print w
	for i in range(trainNum):
		w+=a[i]*y_train[i]*x_train[i]

	# print w

	w0 = 0
	for i in range(trainNum):
		w0+= y_train[i]+np.dot(w.T, x_train[i]);

	w0 = w0/trainNum;
	# print w0

	slop = -w[0] / w[1]
	xx = np.linspace(-1.5, 1.5)
	yy = slop*xx - (w0)/w[1]


	plt.plot(xx, yy, 'k-')
	def f(x, w, b, c=0):
		return (-w[0] * x - b + c) / w[1]
	a0 = -1.2; a1 = f(a0, w, w0, 1)
	b0 = 1.2; b1 = f(b0, w, w0, 1)
	plt.plot([a0,b0], [a1,b1], "k--")

	a0 = -1.2; a1 = f(a0, w, w0, -1)
	b0 = 1.2; b1 = f(b0, w, w0, -1)
	plt.plot([a0,b0], [a1,b1], "k--")
def SVM_SK_Kernel(gen, testNum):
	print gen.shape
	num, col = gen.shape
	x = gen[:,:2]
	y = gen[:,-1:]

	x_train = gen[:num - testNum,:2]
	y_train = gen[:num - testNum,-1:]

	x_test = gen[-testNum:, :2]
	y_test = gen[-testNum:, -1:]
	fold = 2

	# x_train, y_train = crossValidMatrix(x, y, fold, 1);
	# x_test, y_test = crossValidMatrix(x,y,fold,0);

	# print x_train.shape, y_train.shape, x_test.shape, y_test.shape
	# clf = svm.SVC(kernel='linear',C=1)
	clf = svm.SVC()
	clf.fit(x_train, y_train)
	# print clf
	y_predict = clf.predict(x_test)

	getPerformance(y_predict, y_test, num/fold);

	plt.axis('tight')

	x_min=x[:,0].min()
	x_max=x[:,0].max()
	y_min=x[:,1].min()
	y_max=x[:,1].max()

	xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
	z=clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

	z = z.reshape(xx.shape)
	# plt.pcolormesh(xx,yy,z>0,cmap=plt.cm.Paired)
	plt.contour(xx,yy,z,colors=['k','k','k'],linestyles=['--','-','--'],levels=[-.5,0,.5])
	plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1],s=80, facecolors='none',zorder=10)
	
	plt.axis([-2, 2, -2, 2])
def plotDataKernel(gen):
	num, col = gen.shape
	for i in range(num):
		if gen[i][2] == 1:
			plt.plot(gen[i][0], gen[i][1],'ro')
		else:
			plt.plot(gen[i][0], gen[i][1],'b*')

def getPerformance(y_predict, y, dataNum):
	TP1=0; TP2=0;
	FN1=0; FN2=0;
	FP1=0; FP2=0; 
	TN1=0; TN2=0
	# print y;
	for i in range(0, dataNum):
		if y_predict[i]==1 and y[i] == 1:
			TP1+=1
		elif y_predict[i] == -1 and y[i] == -1:
			TP2+=1


	for i in range(0, dataNum):
		if y_predict[i] != 1 and y[i] != 1:
			TN1+=1
		elif y_predict[i] != -1 and y[i] != -1:
			TN2+=1


	for i in range(0, dataNum):
		if y_predict[i] != 1 and y[i] == 1:
			FN1+=1
		elif y_predict[i] != -1 and y[i] == -1:
			FN2+=1


	for i in range(0, dataNum):
		if y_predict[i] == 1 and y[i] != 1:
			FP1+=1
		elif y_predict[i] == -1 and y[i] != -1:
			FP2+=1


	print "TP: ", TP1, TP2
	print "FN: ",FN1, FN2
	print "FP: ",FP1, FP2
	print "TN: ", TN1, TN2

	print "----- confusion_matrix: -----"
	print confusion_matrix(y, y_predict);
	print "-----------------------------"
	MeasureFunction("1", TP1, FP1, FN1, TN1, 1);
	MeasureFunction("-1", TP2, FP2, FN2, TN2, 1);

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

def main():
	# generateLSdData()
	# generateNonLSdData()

	# gen = loadData("dataFileNonLS.txt")
	# gen = loadData("dataFile.txt")
	# gen = loadData("dataFileKernel.txt")
	# gen = loadData("dataFile2.txt")
	gen = loadData("dataFileKernel_SubExm.txt")
	num, col = gen.shape

	# SVM_SK_Linear(gen, testNum = 0)
	# SVM_SK_Linear(gen, testNum = num/5)
	SVM_SK_Kernel(gen, testNum = 0)
	# SVM_SK_Kernel(gen, testNum = num/5)
	# SVM(gen,testNum = num/5)
	# SVM(gen,testNum = 0)
	# SVM_Kernel(gen, testNum=0)
	# SVM_Kernel(gen, testNum=num/5)

	print "------- total performance ------";
	MeasureFunction("total",31,25,18,25,1)

	# plotData(gen)
	plotDataKernel(gen)
	plt.show()
if __name__ == "__main__":
	main()