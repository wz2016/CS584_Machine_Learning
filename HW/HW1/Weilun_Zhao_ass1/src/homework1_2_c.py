import numpy as np
import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

a1 =np.loadtxt('mvar-set1.dat')
A1 = a1[:]

a2 =np.loadtxt('mvar-set2.dat')
A2 = a2[:]

a3 =np.loadtxt('mvar-set3.dat')
A3 = a3[:]

a4 =np.loadtxt('mvar-set4.dat')
A4 = a4[:]

def ExplicitSolve(A):
	tot = A.size
	n = A[:,[0]].size
	m = tot/n
	# print m
	xdata = np.mat(A[:-20, 0:m-1])
	ydata = np.mat(A[:-20, [m-1]])
	# print xdata.shape
	# print ydata.shape
	coef = ((xdata.T*xdata).I)*(xdata.T)*ydata
	# print coef.shape
	# print coef
	return coef

param1 = ExplicitSolve(A1)

param2 = ExplicitSolve(A2)

param3 = ExplicitSolve(A3)

param4 = ExplicitSolve(A4)


def MSE(A, param):
	tot = A.size
	n = A[:,[0]].size
	m = tot/n
	# print m
	xdata_train = np.mat(A[:-20, 0:m-1])
	ydata_train = np.mat(A[:-20, [m-1]])

	j = ((xdata_train*param - ydata_train).T)*(xdata_train*param - ydata_train)
	# print j
	mse = j/(n-20)
	print "train mse: ", mse
	# return mse
	xdata_test = np.mat(A[-20:, 0:m-1])
	ydata_test = np.mat(A[-20:, [m-1]])

	j_test = ((xdata_test*param - ydata_test).T)*(xdata_test*param - ydata_test)
	mse_test = j_test/(20)
	print "test mse: ", mse
	# print j
# MSE(A1, param1)
# MSE(A2, param2)
# MSE(A3, param3)
MSE(A4, param4)