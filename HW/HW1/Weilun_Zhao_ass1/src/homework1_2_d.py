import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import norm

a1 =np.loadtxt('mvar-set1.dat')
A1 = a1[:]

a2 =np.loadtxt('mvar-set2.dat')
A2 = a2[:]

a3 =np.loadtxt('mvar-set3.dat')
A3 = a3[:]

a4 =np.loadtxt('mvar-set4.dat')
A4 = a4[:]
#
def findMatrixLoc(A):
	# print "median:", float(np.median(A))
	return float(np.median(A))

def findMatrixScale(A):
	min = float(A.min())
	max = float(A.max())
	scale = (max-min)/2. 
	return scale

# def GaussianExpAndV(A,loc, scale):
# 	return norm.fit(A,loc =loc, scale=scale)

def GaussianKernel(X1, X2, sigma = 0.2):
	return np.exp(-((X1-X2)**2)/(2*sigma))

# x= np.array([0.5, 0.25])

# print findMatrixLoc(x)
# print findMatrixScale(x)
def MSE(G, X, Y):
	total_G = G.size
	n_G = G[0].size
	m_G = total_G/n_G
	# print n_G
	n = X[:,[0]].size
	G_mat = np.mat(G)
	X_mat = np.mat(X)
	GX = np.array([np.dot(X_mat,G_mat)])
	# print GX.shape
	# print n, n_G
	sumTotal = 0
	# print GX[0][0][0]
	for i in range (0, n):
		sumlocal = 0
		for j in range(0, n_G):
			sumlocal+=GX[0][i][j]
		sumTotal+= (sumlocal-Y[i])**2

	print 'MES: ', sumlocal/n

def run(A):
	tot = A.size
	n = A[:,[0]].size
	m = tot/n

	xdata_train = np.mat(A[:-20, 0:m-1])
	ydata_train = np.mat(A[:-20, [m-1]])

	G = np.zeros((m-1,m-1))
	X = np.zeros((m-1,2))
	for i in range(0, m-1):
		X[i][0] = findMatrixLoc(xdata_train[:-20,[i]])
		X[i][1] = findMatrixScale(xdata_train[:-20,[i]])
		# print 'loc and scale',X[i]

	XExpAndV = np.zeros((m-1,2))
	for i in range(0, m-1):
		loc = X[i][0]
		scale = X[i][1]
		XExpAndV[i] = np.array(norm.fit(xdata_train[:,[i]],loc=loc, scale = scale ))
		# print 'exp and v', XExpAndV[i]

	for i in range(0, m-1):
		for j in range(0, m-1):
			G[i][j] = GaussianKernel(XExpAndV[i][0], XExpAndV[j][0], (XExpAndV[i][0]+ XExpAndV[i][1])/2)
	
	# print G
	print "train MSE: "
	MSE(G, xdata_train, ydata_train)

	xdata_test = np.mat(A[-20:, 0:m-1])
	ydata_test = np.mat(A[-20:, [m-1]])
	print "test MSE: "
	MSE(G, xdata_test, ydata_test)



run(A1)
run(A2)
run(A3)
run(A4)
