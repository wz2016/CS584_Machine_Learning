import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import norm

a1 =np.loadtxt('mvar-set1.dat')
A1 = a1[:]

# a2 =np.loadtxt('mvar-set2.dat')
# A2 = a2[:]

# a3 =np.loadtxt('mvar-set3.dat')
# A3 = a3[:]

# a4 =np.loadtxt('mvar-set4.dat')
# A4 = a4[:]

def getArph(A):
	tot = A.size
	n = A[:, [0]].size
	m = tot/n
	n_train = n-20
	# print m
	xdata_train = A[:-20, 0:m-1]
	ydata_train = A[:-20, [m-1]]
	X = np.matrix(xdata_train)
	Y = np.matrix(ydata_train)
	G = np.zeros((n_train,n_train))
	for i in range(0, n_train):
		for j in range(0, n_train):
			G[i][j] = GaussianKernel(X[i],X[j],0.2)

	# print G.shape
	G = np.mat(G)
	arph = (G.I)*ydata_train
	print arph.shape
	return arph

def GaussianKernel(x, y, s):
	sigma = s
	numerator = 0
	# print x.size

	for i in range(0, x.size):
		numerator+=(x[0,i]-y[0,i])**2;
	denomiator = (-2)*(sigma**2)
	return np.exp(numerator/denomiator)

def MSE(A):
	tot = A.size
	n = A[:, [0]].size
	m = tot/n

	n_train = n-20
	n_test = 20
	# print m
	xdata_train = A[:-20, 0:m-1]
	ydata_train = A[:-20, [m-1]]

	xdata_test = A[-20:, 0:m-1]
	ydata_test = A[-20:, [m-1]]

	arph = getArph(A)
	RSS = 0
	for i in range(0, n_train):
		sumExpect = 0
		for j in range(0, n_train):
			sumExpect += arph[j]*GaussianKernel(xdata_train[j], xdata_train[i],0.2)
		RSS += (sumExpect-ydata_train[i])**2

	MSE = RSS/n_train
	print "MSE train: ", MSE

	RSS = 0
	for i in range(0, n_test):
		sumExpect = 0
		for j in range(0, n_train):
			sumExpect += arph[j]*GaussianKernel(xdata_train[j], xdata_test[i],0.2)
		RSS += (sumExpect-ydata_test[i])**2

	MSE = RSS/n_test
	print "MSE test: ", MSE

MSE(A1)

# MSE(A2)

# MSE(A3)

# MSE(A4)

