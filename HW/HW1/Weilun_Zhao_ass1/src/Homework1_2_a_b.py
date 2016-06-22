import numpy as np
import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def MSE(a, arr):
	tot = a.size
	n = a[:,[0]].size
	m = tot/n
	i =0
	j = 0
	# print arr.size
	# print arr[0]
	sumyield = 0
	for i in range(0,n):
		sumlocal = arr[0]
		for j in range(1, m):
			sumlocal += arr[j]*a[i][j-1]
			# print i, j, a[i][j-1]
			# print i,j,arr[j]
		sumyield += (sumlocal - a[i][m-1])**2
	print "MSE: ", sumyield/n

def data_Matrix(a):
	tot = a.size
	# print tot
	m = a[:,[1]].size
	print "Row: ", m
	n = tot/m
	# print n
	xdata = a[:, 0:n-1]
	ydata = a[:, [n-1]]
	# 
	# x0x0 = a[:,[0]]*a[:,[0]]
	# x1x1 = a[:,[0]]*a[:,[1]]
	# x0x1 = a[:,[1]]*a[:,[0]]
	# xdata = np.append(xdata,x0x0,axis=1)
	# xdata= np.append(xdata,x0x1,axis=1)
	# xdata= np.append(xdata,x1x1,axis=1)
	# xdata= np.append(xdata,ydata,axis=1)

	# arr1 = np.array([1.0223, 0.9975, 0.9905, -0.0126, -0.0085, -0.0064])
	# arr2 = np.array([0.0004,0.0646, -0.0006, 2.058e-05, -0.0011, 0.0003])
	arr3 = np.array([0.9965, 0.9980, 1.0008, 0.9988, -0.0011, 1.9998, 0.0014, -0.0013, -0.0005, -0.0003, 0.0009])
	# arr4 = np.array([0.0440, 5.909e-05, 0.0001, 0.0001, 0.0002, -1.463e-05, -0.0052,-0.0001, -0.0050, -0.0049, -0.0051])
	
	# 
	x0x0 = a[:,[0]]*a[:,[0]]
	x1x1 = a[:,[0]]*a[:,[1]]
	x2x2 = a[:,[2]]*a[:,[2]]
	x3x3 = a[:,[3]]*a[:,[3]]
	x4x4 = a[:,[4]]*a[:,[4]]

	xdata = np.append(xdata,x0x0,axis=1)
	xdata= np.append(xdata,x1x1,axis=1)
	xdata= np.append(xdata,x2x2,axis=1)
	xdata= np.append(xdata,x3x3,axis=1)
	xdata= np.append(xdata,x4x4,axis=1)
	# xdata = np.append(xdata, ydata, axis = 1)
	# print xdata.shape
	# 
	# MSE(xdata, arr1)
	# MSE(xdata, arr2)
	MSE(xdata, arr3)
	# MSE(xdata, arr4)
	# data_Anaylsis(xdata, ydata)

def data_Anaylsis(xdata, ydata):
	x = sm.add_constant(xdata)
	est = sm.OLS(ydata, x).fit()
	print est.summary()

a1 =np.loadtxt('mvar-set1.dat')
A1 = a1[:]

a2 =np.loadtxt('mvar-set2.dat')
A2 = a2[:]

a3 =np.loadtxt('mvar-set3.dat')
A3 = a3[-100:]

a4 =np.loadtxt('mvar-set4.dat')
A4 = a4[:]

# data_Matrix(A1)

# data_Matrix(A2)

data_Matrix(A3)

# data_Matrix(A4)


	
# arr1 = np.array([0.9959,0.9975,0.9905])
# arr2 = np.array([0.0009,0.0646,-0.0006])
# arr3 = np.array([0.9989,0.9979,1.0008, 0.9988, -0.0012, 1.9998])
# arr4 = np.array([0.0101,4.668e-05,0.0001, 3.952e-05, 0.0002, -1.837e-06])


# MSE(A1, arr1)
# MSE(A2, arr2)
# MSE(A3, arr3)
# MSE(A4, arr4)